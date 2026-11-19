///|
test "delta memory decays before the new value correction" {
  let state = @gated_delta_reference.State::new({
    key_heads: 1,
    key_width: 1,
    value_heads: 1,
    value_width: 1,
    convolution_width: 4,
  })
  let out = FixedArray::make(1, 0.0F)
  state.run_into([1.0F], [1.0F], [2.0F], [0.0F], [0.5F], out)
  assert_true((out[0] - 1.0F).abs() < 0.000002F)
  state.run_into([1.0F], [1.0F], [4.0F], [-0.6931472F], [0.5F], out)
  assert_true((out[0] - 2.25F).abs() < 0.00001F)
  // Model decay is not clipped to the Kimi-Delta reference's [-5,0] range.
  state.run_into([1.0F], [1.0F], [0.0F], [-10.0F], [0.0F], out)
  assert_true(out[0] > 0.0F && out[0] < 0.00011F)
  state.run_into([1.0F], [1.0F], [0.0F], [-1.0F / 0.0F], [0.0F], out)
  assert_eq(out[0], 0.0F)
}

///|
test "grouped qk heads support non-square state and explicit continuation" {
  let g : @hybrid.RecurrentAttention = {
    key_heads: 1,
    key_width: 2,
    value_heads: 2,
    value_width: 3,
    convolution_width: 4,
  }
  let state = @gated_delta_reference.State::new(g)
  let out = FixedArray::make(6, 0.0F)
  state.run_into(
    [0.0F, 1.0F],
    [0.0F, 1.0F],
    [2.0F, 4.0F, 6.0F, 10.0F, 20.0F, 30.0F],
    [0.0F, 0.0F],
    [1.0F, 0.5F],
    out,
  )
  for
    i, expected in [
      1.414213F, 2.828426F, 4.242637F, 3.535531F, 7.071062F, 10.606592F,
    ] {
    assert_true((out[i] - expected).abs() < 0.00003F)
  }
  let checkpoint = FixedArray::make(state.state_elements(), 0.0F)
  state.copy_into(checkpoint)
  let restored = @gated_delta_reference.State::new(g)
  restored.restore(checkpoint[:])
  let other = FixedArray::make(6, 0.0F)
  state.run_into(
    [1.0F, 1.0F],
    [1.0F, 0.0F],
    [1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F],
    [-1.0F, -2.0F],
    [0.7F, 0.8F],
    out,
  )
  restored.run_into(
    [1.0F, 1.0F],
    [1.0F, 0.0F],
    [1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F],
    [-1.0F, -2.0F],
    [0.7F, 0.8F],
    other,
  )
  for i in 0..<6 {
    assert_eq(out[i], other[i])
  }
  restored.reset()
  restored.copy_into(checkpoint)
  assert_true(checkpoint.iter().all(x => x == 0.0F))
}
