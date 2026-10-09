///|
test "KV assembly broadcasts shared suffix and extracts strided values" {
  let limits = try! GlmHybridReferenceLimits::new(
    maximum_sequence_length=4,
    maximum_head_count=4,
    maximum_head_dimension=8,
    maximum_channel_count=16,
    maximum_vision_patch_count=4,
  )
  let full = try! dsa_kv_attention_assembly(
    [1.0, 2.0, 10.0, 3.0, 4.0, 20.0],
    [9.0],
    row_count=1,
    head_count=2,
    key_nope_width=2,
    shared_key_suffix_width=1,
    value_width=1,
    limits~,
  )
  assert_eq(full.keys(), [1.0, 2.0, 9.0, 3.0, 4.0, 9.0])
  assert_eq(full.values(), [10.0, 20.0])
  let flash = try! dsa_kv_attention_assembly(
    [1.0, 2.0, 10.0, 3.0, 4.0, 20.0],
    [],
    row_count=1,
    head_count=2,
    key_nope_width=2,
    shared_key_suffix_width=0,
    value_width=1,
    limits~,
  )
  assert_eq(flash.keys(), [1.0, 2.0, 3.0, 4.0])
  assert_eq(flash.values(), [10.0, 20.0])
}

///|
test "KV assembly rejects nonempty nonexistent suffix" {
  let limits = try! GlmHybridReferenceLimits::new(
    maximum_sequence_length=4,
    maximum_head_count=4,
    maximum_head_dimension=8,
    maximum_channel_count=16,
    maximum_vision_patch_count=4,
  )
  try
    dsa_kv_attention_assembly(
      [1.0, 2.0],
      [3.0],
      row_count=1,
      head_count=1,
      key_nope_width=1,
      shared_key_suffix_width=0,
      value_width=1,
      limits~,
    )
  catch {
    InvalidShape => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("Flash suffix accepted")
  }
}
