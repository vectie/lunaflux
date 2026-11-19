///|
/// Offline bit-pattern oracle. UInt carries a raw 16-bit BF16 word so signed
/// zero and NaN payloads survive; this is not a production CPU backend.
pub fn pack_qkv_words_into(
  plan : @precision.PackedQkvPrecision,
  query : ArrayView[UInt32],
  key : ArrayView[UInt32],
  value : ArrayView[UInt32],
  output : FixedArray[UInt32],
  rows~ : Int32,
) -> Unit raise {
  let (q, kv, maximum) = plan.geometry()
  let width = q + kv * 2
  guard rows > 0 &&
    rows <= maximum &&
    query.length().to_int64() == rows.to_int64() * q.to_int64() &&
    key.length().to_int64() == rows.to_int64() * kv.to_int64() &&
    value.length() == key.length() &&
    output.length().to_int64() == rows.to_int64() * width.to_int64() &&
    query.all(word => word <= 65535U) &&
    key.all(word => word <= 65535U) &&
    value.all(word => word <= 65535U) else {
    fail("invalid BF16 QKV words")
  }
  for row in 0..<rows {
    for column in 0..<q {
      output[row * width + column] = query[row * q + column]
    }
    for column in 0..<kv {
      output[row * width + q + column] = key[row * kv + column]
      output[row * width + q + kv + column] = value[row * kv + column]
    }
  }
}

///|
/// Offline split-plane scatter oracle. Caller supplies valid CSR rows and
/// uniquely owned writable pages, as for the existing device KV store program.
pub fn paged_kv_words_into(
  program : @effects.KvWriteProgram,
  packed : ArrayView[UInt32],
  positions : ArrayView[Int32],
  row_offsets : ArrayView[Int32],
  page_offsets : ArrayView[Int32],
  physical_pages : ArrayView[Int32],
  key : FixedArray[UInt32],
  value : FixedArray[UInt32],
  tokens_per_page~ : Int32,
) -> Unit raise {
  let (q, kv, stride) = program.geometry()
  let width = q + kv * 2
  let rows = row_offsets.length() - 1
  guard tokens_per_page > 0 &&
    rows > 0 &&
    page_offsets.length() == rows + 1 &&
    row_offsets[0] == 0 &&
    row_offsets[rows] == positions.length() &&
    packed.length().to_int64() ==
    positions.length().to_int64() * width.to_int64() &&
    stride >= tokens_per_page.to_int64() * kv.to_int64() &&
    stride <= 2147483647L &&
    key.length() == value.length() &&
    key.length().to_int64() % stride == 0L else {
    fail("invalid paged KV buffers")
  }
  for row in 0..<rows {
    guard row_offsets[row] <= row_offsets[row + 1] &&
      row_offsets[row] >= 0 &&
      row_offsets[row + 1] <= positions.length() else {
      fail("invalid KV CSR row")
    }
    for token in row_offsets[row]..<row_offsets[row + 1] {
      let position = positions[token]
      guard position >= 0 else { fail("invalid KV position") }
      let table = page_offsets[row].to_int64() +
        (position / tokens_per_page).to_int64()
      guard table >= 0L &&
        table >= page_offsets[row].to_int64() &&
        table < page_offsets[row + 1].to_int64() &&
        table < physical_pages.length().to_int64() else {
        fail("invalid KV page table")
      }
      let page = physical_pages[table.to_int()]
      guard page >= 0 && page.to_int64() < key.length().to_int64() / stride else {
        fail("invalid KV physical page")
      }
      let destination = (page.to_int64() * stride +
      (position % tokens_per_page).to_int64() * kv.to_int64()).to_int()
      for column in 0..<kv {
        key[destination + column] = packed[token * width + q + column]
        value[destination + column] = packed[token * width + q + kv + column]
      }
    }
  }
}
