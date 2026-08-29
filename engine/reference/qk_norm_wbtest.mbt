///|
fn qk_test_limits() -> @refkernels.ReferenceLimits {
  try! @refkernels.ReferenceLimits::new(
    max_rows=2,
    max_columns=16,
    max_elements=32,
  )
}

///|
test "QK RMS norm is per head and copies packed V exactly" {
  let limits = qk_test_limits()
  let input = try! @refkernels.Activation::new(
    rows=1,
    columns=8,
    values=[3.0F, 4.0F, 0.0F, 2.0F, 6.0F, 8.0F, 9.0F, 10.0F],
    limits~,
  )
  let query_weight = try! @refkernels.Tensor::new(
    rows=1,
    columns=2,
    values=[1.0F, 2.0F],
    limits~,
  )
  let key_weight = try! @refkernels.Tensor::new(
    rows=1,
    columns=2,
    values=[0.5F, 1.5F],
    limits~,
  )
  let output = try! qk_rms_norm(
    input,
    query_weight,
    key_weight,
    query_heads=2,
    kv_heads=1,
    head_dimension=2,
    epsilon=0.000001F,
    limits~,
  )
  assert_true((output.at(row=0, column=0) - 0.8485281F).abs() < 0.000001F)
  assert_true((output.at(row=0, column=1) - 2.2627416F).abs() < 0.000001F)
  assert_true((output.at(row=0, column=2) - 0.0F).abs() < 0.000001F)
  assert_true((output.at(row=0, column=3) - 2.8284264F).abs() < 0.000001F)
  assert_true((output.at(row=0, column=4) - 0.4242640F).abs() < 0.000001F)
  assert_true((output.at(row=0, column=5) - 1.6970562F).abs() < 0.000001F)
  assert_eq(output.at(row=0, column=6), 9.0F)
  assert_eq(output.at(row=0, column=7), 10.0F)
}

///|
test "QK RMS norm rejects packed or weight shape substitution" {
  let limits = qk_test_limits()
  let input = try! @refkernels.Activation::new(
    rows=1,
    columns=8,
    values=[0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F],
    limits~,
  )
  let weight = try! @refkernels.Tensor::new(
    rows=1,
    columns=2,
    values=[1.0F, 1.0F],
    limits~,
  )
  try
    qk_rms_norm(
      input,
      weight,
      weight,
      query_heads=1,
      kv_heads=1,
      head_dimension=2,
      epsilon=0.000001F,
      limits~,
    )
  catch {
    InvalidShape(NormalizationWeight) => ()
    _ => fail("unexpected packed-width rejection")
  } noraise {
    _ => fail("substituted packed QKV width was admitted")
  }
}
