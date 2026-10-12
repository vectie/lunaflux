///|
/// Scalar pieces of attention; matrix reduction and probability storage remain
/// their own typed laws. A physical matrix fragment is not an ordered fold.
pub(all) enum AttentionExpression {
  DotUpdate
  ScaledScore
  Maximum
  Exponential
  SumUpdate
  Rescale
  ProbabilityStorage
  OnlineTotal
  ValueUpdate
  SinkTotal
  Publish
}

///|
pub fn AttentionProbabilityLaw::expression(
  self : Self,
  operation : AttentionExpression,
  accumulation~ : AttentionAccumulationLaw,
) -> NumericExpression raise PrecisionPlanError {
  self.validate()
  let x = Input(0)
  let y = Input(1)
  let z = Input(2)
  let exponential = Binary(SubtractRN, x, y)
  match operation {
    DotUpdate =>
      numeric_expression(
        [Float32, BFloat16, BFloat16],
        Binary(
          AddRN,
          x,
          Binary(MultiplyRN, Convert(Float32, y), Convert(Float32, z)),
        ),
      )
    ScaledScore | Rescale =>
      numeric_expression([Float32, Float32], Binary(MultiplyRN, x, y))
    Maximum =>
      numeric_expression([Float32, Float32], Binary(MaximumNumber, x, y))
    Exponential =>
      numeric_expression([Float32, Float32], Unary(NaturalExp, exponential))
    SumUpdate => numeric_expression([Float32, Float32], Binary(AddRN, x, y))
    ProbabilityStorage =>
      numeric_expression(
        [Float32],
        match self {
          OnlineBf16(_) | GlobalUnnormalizedBf16 =>
            Convert(Float32, Convert(BFloat16, x))
          _ => x
        },
      )
    OnlineTotal =>
      numeric_expression(
        [Float32, Float32, Float32],
        match accumulation {
          TiledFloat32 => FusedMultiplyAdd(x, y, z)
          OrderedFloat32 => Binary(AddRN, Binary(MultiplyRN, x, y), z)
        },
      )
    ValueUpdate =>
      numeric_expression(
        [Float32, Float32, BFloat16],
        match self {
          OnlineBf16(_) => FusedMultiplyAdd(y, Convert(Float32, z), x)
          _ => Binary(AddRN, x, Binary(MultiplyRN, y, Convert(Float32, z)))
        },
      )
    SinkTotal =>
      numeric_expression(
        [Float32, Float32, Float32],
        Binary(AddRN, x, Unary(NaturalExp, Binary(SubtractRN, y, z))),
      )
    Publish =>
      numeric_expression(
        [Float32, Float32],
        Convert(BFloat16, Binary(DivideRN, x, y)),
      )
  }
}
