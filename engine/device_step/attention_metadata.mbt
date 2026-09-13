///|
fn attention_metadata_layout(
  limits : DeviceStepLimits,
) -> @luna_attention_metadata.AttentionMetadataLayout {
  @luna_attention_metadata.AttentionMetadataLayout::new(
    maximum_rows=Int::min(limits.max_rows(), limits.max_tokens()),
    maximum_tokens=limits.max_tokens(),
  ).unwrap()
}

///|
/// Materialize once after preflight, reusing the descriptor's preallocated
/// storage and existing upload. Every head/layer sees the same immutable epoch.
fn publish_attention_tile_metadata(
  host : DeviceStepHostBuffers,
  summary : DeviceStepSummary,
) -> Unit {
  if !host.attention_metadata_enabled {
    return
  }
  let base = host.attention_metadata_offset
  for bucket in 0..<4 {
    let width = 16 << bucket
    let records = base + host.attention_metadata_layout.bucket_offset(bucket)
    let mut count = 0
    for row in 0..<summary.prefill_count {
      let begin = read_descriptor_i32(host.query_row_offsets, row)
      let end = read_descriptor_i32(host.query_row_offsets, row + 1)
      let length = read_descriptor_i32(host.sequence_lengths, row)
      let page_begin = read_descriptor_i32(host.page_offsets, row)
      let page_end = read_descriptor_i32(host.page_offsets, row + 1)
      let mut token = begin
      while token < end {
        let mut key_end = 0
        for query in token..<Int::min(token + width, end) {
          key_end = Int::max(
            key_end,
            read_descriptor_i32(host.token_positions, query) + 1,
          )
        }
        let record = records + count * 8
        write_i32(host.counts, record, row)
        write_i32(host.counts, record + 1, token)
        write_i32(host.counts, record + 2, begin)
        write_i32(host.counts, record + 3, end)
        write_i32(host.counts, record + 4, length)
        write_i32(host.counts, record + 5, page_begin)
        write_i32(host.counts, record + 6, page_end)
        write_i32(host.counts, record + 7, key_end)
        count += 1
        token += width
      }
    }
    write_i32(host.counts, base + bucket, count)
  }
}

///|
fn attention_metadata_argument(
  descriptor : DeviceStepResources,
  profile : @launch_contract.PagedKernelProfile,
) -> @device.KernelArgument raise DeviceStepError {
  let offset = OUTPUT_ROW_ENDS_CELL + profile.max_query_rows() + 1
  let layout = @luna_attention_metadata.AttentionMetadataLayout::new(
    maximum_rows=Int::min(profile.max_query_rows(), profile.max_query_tokens()),
    maximum_tokens=profile.max_query_tokens(),
  ).unwrap()
  @device.KernelArgument::device_region(
    descriptor.counts,
    offset=byte_capacity(offset),
    byte_count=byte_capacity(layout.cell_count()),
    alignment=4L,
  ) catch {
    error => raise ExecutorDevice(KernelArguments, error)
  }
}
