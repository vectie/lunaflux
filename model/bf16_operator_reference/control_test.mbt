///|
test "control uses thresholded softplus, per-head bias and unbounded decay" {
  let plan = @precision.GatedDeltaControlPrecision::new(
    head_count=2,
    maximum_rows=2,
  )
  let decay = FixedArray::make(4, 0.0F)
  let beta = FixedArray::make(4, 0.0F)
  @bf16_operator_reference.gated_delta_control_into(
    plan,
    [0.0F, 21.0F, -100.0F, 100.0F],
    [0.0F, 1.0F, -100.0F, 100.0F],
    [0.0F, 0.0F],
    [0.0F, 1.0F],
    decay,
    beta,
    rows=2,
  )
  assert_true((decay[0] + 0.6931472F).abs() < 0.000001F)
  assert_eq(decay[1], -22.0F)
  assert_true(decay[2] <= 0.0F && decay[2] > -0.000001F)
  assert_eq(decay[3], -101.0F)
  assert_eq(beta[:].to_owned(), [0.5F, 0.73046875F, 0.0F, 1.0F])
}
