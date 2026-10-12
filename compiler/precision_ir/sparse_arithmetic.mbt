///|
/// Existing selected-index and pooling numerical program. Estimate division is
/// explicit: migrating this program must not silently upgrade or relax it.
pub(all) enum SparseExpression {
  DotUpdate
  Score
  DenominatorUpdate
  Probability
  Publish
  PoolLogit
  ReluScore
  WeightedScore
}

///|
pub fn SparseExpression::expression(
  self : Self,
) -> NumericExpression raise PrecisionPlanError {
  let x = Input(0)
  let y = Input(1)
  let z = Input(2)
  match self {
    DotUpdate =>
      numeric_expression(
        [Float32, BFloat16, BFloat16],
        Binary(
          AddRN,
          x,
          Binary(MultiplyRN, Convert(Float32, y), Convert(Float32, z)),
        ),
      )
    Score => numeric_expression([Float32, Float32], Binary(MultiplyRN, x, y))
    DenominatorUpdate =>
      numeric_expression(
        [Float32, Float32, Float32],
        Binary(AddRN, x, Unary(NaturalExp, Binary(SubtractRN, y, z))),
      )
    Probability =>
      numeric_expression(
        [Float32, Float32, Float32],
        Convert(
          BFloat16,
          Binary(DivideEstimate, Unary(NaturalExp, Binary(SubtractRN, x, y)), z),
        ),
      )
    Publish => numeric_expression([Float32], Convert(BFloat16, x))
    PoolLogit =>
      numeric_expression(
        [BFloat16, BFloat16],
        Binary(AddRN, Convert(Float32, x), Convert(Float32, y)),
      )
    ReluScore =>
      numeric_expression(
        [Float32, Float32],
        Binary(MaximumNumber, Binary(MultiplyRN, x, y), Float32Bits(0U)),
      )
    WeightedScore =>
      numeric_expression(
        [Float32, BFloat16, Float32],
        Binary(MultiplyRN, x, Binary(MultiplyRN, Convert(Float32, y), z)),
      )
  }
}
