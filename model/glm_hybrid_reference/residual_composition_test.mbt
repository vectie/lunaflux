///|
test "outer residual composition preserves plain and transposed hyper ordering" {
  let limits = try! GlmHybridReferenceLimits::new(
    maximum_sequence_length=4,
    maximum_head_count=4,
    maximum_head_dimension=4,
    maximum_channel_count=8,
    maximum_vision_patch_count=4,
  )
  let plain = try! moe_decoder_residual_add(
    [1.0, 2.0],
    [3.0, 4.0],
    row_count=1,
    hidden_width=2,
    limits~,
  )
  assert_eq(plain.values(), [4.0, 6.0])
  let mixed = try! hyper_connection_residual_mix(
    [2.0, 4.0],
    [1.0, 2.0, 3.0, 4.0],
    [10.0, 20.0],
    [3.0],
    row_count=1,
    stream_count=2,
    hidden_width=1,
    limits~,
  )
  assert_eq(mixed.values(), [76.0, 112.0])
}

///|
test "hyper composition rejects mismatched controls" {
  let limits = try! GlmHybridReferenceLimits::new(
    maximum_sequence_length=4,
    maximum_head_count=4,
    maximum_head_dimension=4,
    maximum_channel_count=8,
    maximum_vision_patch_count=4,
  )
  try
    hyper_connection_residual_mix(
      [1.0],
      [1.0],
      [1.0, 2.0],
      [1.0],
      row_count=1,
      stream_count=2,
      hidden_width=1,
      limits~,
    )
  catch {
    InvalidShape => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("mismatched controls accepted")
  }
}
