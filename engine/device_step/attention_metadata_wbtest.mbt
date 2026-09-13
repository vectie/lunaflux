///|
test "step metadata covers ragged prefill once and excludes decode on every tile width" {
  let host = make_host_buffers(step_device_limits(max_rows=8, max_tokens=256))
  host.attention_metadata_enabled = true
  for index, value in [0, 1, 34, 163, 164] {
    write_i32(host.query_row_offsets, index, value)
  }
  for index, value in [0, 1, 6, 24, 25] {
    write_i32(host.page_offsets, index, value)
  }
  for row, length in [7, 39, 135, 8] {
    write_i32(host.sequence_lengths, row, length)
    let begin = read_descriptor_i32(host.query_row_offsets, row)
    let end = read_descriptor_i32(host.query_row_offsets, row + 1)
    for token in begin..<end {
      write_i32(
        host.token_positions,
        token,
        length - (end - begin) + token - begin,
      )
    }
  }
  let summary : DeviceStepSummary = {
    sequence_value: 1UL,
    prefill_count: 3,
    decode_count: 1,
    row_count: 4,
    token_count: 164,
    page_count: 25,
  }
  publish_attention_tile_metadata(host, summary)
  for bucket in 0..<4 {
    let width = 16 << bucket
    let count = read_descriptor_i32(
      host.counts,
      host.attention_metadata_offset + bucket,
    )
    let records = host.attention_metadata_offset +
      host.attention_metadata_layout.bucket_offset(bucket)
    let covered = FixedArray::make(164, 0)
    for tile in 0..<count {
      let record = records + tile * 8
      let row = read_descriptor_i32(host.counts, record)
      let begin = read_descriptor_i32(host.counts, record + 1)
      let end = Int::min(
        begin + width,
        read_descriptor_i32(host.counts, record + 3),
      )
      assert_true(row < 3)
      assert_eq(
        read_descriptor_i32(host.counts, record + 7),
        read_descriptor_i32(host.token_positions, end - 1) + 1,
      )
      for token in begin..<end {
        covered[token] += 1
      }
    }
    for token in 0..<163 {
      assert_eq(covered[token], 1)
    }
    assert_eq(covered[163], 0)
  }
  assert_eq(uploaded_count_cells(host, summary) * 4, host.counts.length())
  publish_attention_tile_metadata(host, empty_device_step_summary())
  for bucket in 0..<4 {
    assert_eq(
      read_descriptor_i32(host.counts, host.attention_metadata_offset + bucket),
      0,
    )
  }
}
