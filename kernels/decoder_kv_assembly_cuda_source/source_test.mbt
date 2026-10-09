///|
test "KV assembly emits Full shared suffix broadcast and strided value split" {
  let rendered = decoder_kv_assembly_bf16_source(
    head_count=64,
    key_nope_width=192,
    shared_key_suffix_width=64,
    value_width=256,
    maximum_rows=32,
    function_symbol="assemble_full",
  )
  let source = @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("const __nv_bfloat16 *shared_key_suffix"))
  assert_true(source.contains("row * 64ULL + component - 192ULL"))
  assert_true(source.contains("head_row * 448ULL + 192ULL + component"))
  assert_true(source.contains("keys == values || keys == packed_kv"))
  assert_false(source.contains("__bfloat162float"))
}

///|
test "KV assembly omits nonexistent Flash suffix operand" {
  let rendered = decoder_kv_assembly_bf16_source(
    head_count=64,
    key_nope_width=256,
    shared_key_suffix_width=0,
    value_width=256,
    maximum_rows=32,
    function_symbol="assemble_flash",
  )
  let source = @utf8.decode(rendered.source_bytes())
  assert_false(source.contains("shared_key_suffix"))
  assert_true(source.contains("head_row * 512ULL + 256ULL + component"))
}
