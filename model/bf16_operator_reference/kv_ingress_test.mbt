///|
test "QKV packing preserves BF16 payloads and fragmented cache continuation" {
  let pack = @precision.PackedQkvPrecision::new(
    query_width=4,
    kv_width=2,
    maximum_rows=3,
  )
  let packed = FixedArray::make(24, 0U)
  @bf16_operator_reference.pack_qkv_words_into(
    pack,
    [1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U, 11U, 12U],
    [0x8000U, 0x7fc1U, 0x0000U, 0xff81U, 0x3f80U, 0xbf80U],
    [20U, 21U, 22U, 23U, 24U, 25U],
    packed,
    rows=3,
  )
  assert_eq(packed[:].to_owned(), [
    1U, 2U, 3U, 4U, 0x8000U, 0x7fc1U, 20U, 21U, 5U, 6U, 7U, 8U, 0U, 0xff81U, 22U,
    23U, 9U, 10U, 11U, 12U, 0x3f80U, 0xbf80U, 24U, 25U,
  ])
  guard @effects.KvWriteProgram::new(
      query_heads=2,
      kv_heads=1,
      head_dimension=2,
      page_stride_bytes=12L,
      element_bytes=2,
      owners=128,
    )
    is Some(writer) else {
    fail("writer")
  }
  // Two tokens/page, two padding words/page. Row zero continues across a
  // fragmented page boundary; row one starts on its own physical page.
  let key = FixedArray::make(24, 99U)
  let value = FixedArray::make(24, 88U)
  @bf16_operator_reference.paged_kv_words_into(
    writer,
    packed[:],
    [1, 2, 0],
    [0, 2, 3],
    [0, 2, 3],
    [2, 0, 3],
    key,
    value,
    tokens_per_page=2,
  )
  assert_eq(key[:].to_owned(), [
    0U, 0xff81U, 99U, 99U, 99U, 99U, 99U, 99U, 99U, 99U, 99U, 99U, 99U, 99U, 0x8000U,
    0x7fc1U, 99U, 99U, 0x3f80U, 0xbf80U, 99U, 99U, 99U, 99U,
  ])
  assert_eq(value[:].to_owned(), [
    22U, 23U, 88U, 88U, 88U, 88U, 88U, 88U, 88U, 88U, 88U, 88U, 88U, 88U, 20U, 21U,
    88U, 88U, 24U, 25U, 88U, 88U, 88U, 88U,
  ])
}
