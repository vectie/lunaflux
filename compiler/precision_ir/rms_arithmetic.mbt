///|
/// Element program for learned RMS normalization. The reduction tree, width,
/// epsilon and root choice are independent explicit inputs, not CUDA defaults.
pub(all) enum RmsExpression {
  SquareUpdate
  SumUpdate
  Variance
  WeightedPublish
}

///|
pub fn RmsExpression::program(
  self : Self,
) -> NumericExpression raise PrecisionPlanError {
  let x = Input(0)
  let y = Input(1)
  let z = Input(2)
  match self {
    SquareUpdate =>
      numeric_expression(
        [Float32, Float32],
        Binary(AddRN, x, Binary(MultiplyRN, y, y)),
      )
    SumUpdate => numeric_expression([Float32, Float32], Binary(AddRN, x, y))
    Variance =>
      numeric_expression(
        [Float32, Float32, Float32],
        Binary(AddRN, Binary(DivideRN, x, y), z),
      )
    WeightedPublish =>
      numeric_expression(
        [Float32, Float32, Float32],
        Convert(BFloat16, Binary(MultiplyRN, Binary(MultiplyRN, x, y), z)),
      )
  }
}
