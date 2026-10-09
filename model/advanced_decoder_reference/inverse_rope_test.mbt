///|
test "inverse RoPE preserves prefix and conjugates adjacent suffix pairs" {
  let limits = ReferenceLimits::new(
    max_values=64,
    max_width=8,
    max_iterations=4,
  )
  let output = attention_output_inverse_rotary_v1(
    [7.0, 8.0, 1.0, 0.0],
    position=1,
    head_count=1,
    head_dimension=4,
    rope_dimension=2,
    maximum_positions=16,
    original_positions=8,
    compressed_theta=10000.0,
    scaling_factor=1.0,
    beta_fast=32.0,
    beta_slow=1.0,
    limits~,
  )
  assert_eq(output[0], 7.0)
  assert_eq(output[1], 8.0)
  assert_true((output[2] - @math.cosf(1.0F)).abs() < 0.00001F)
  assert_true((output[3] + @math.sinf(1.0F)).abs() < 0.00001F)
}

///|
test "panic inverse RoPE rejects odd suffix and out-of-range position" {
  let limits = ReferenceLimits::new(
    max_values=64,
    max_width=8,
    max_iterations=4,
  )
  try
    attention_output_inverse_rotary_v1(
      [1.0, 2.0, 3.0, 4.0],
      position=16,
      head_count=1,
      head_dimension=4,
      rope_dimension=3,
      maximum_positions=16,
      original_positions=8,
      compressed_theta=10000.0,
      scaling_factor=1.0,
      beta_fast=32.0,
      beta_slow=1.0,
      limits~,
    )
  catch {
    InvalidDimension => ()
    _ => fail("wrong invalid-geometry failure")
  } noraise {
    _ => fail("invalid inverse RoPE geometry was accepted")
  }
}
