///|
/// Projection scalar boundaries; physical tree/MMA schedules retain their
/// explicit reduction laws. No add/multiply contraction is implicit here.
pub(all) enum ProjectionExpression {
  Product
  AddProduct
  Publish
}

///|
pub fn ProjectionExpression::program(
  self : Self,
) -> NumericExpression raise PrecisionPlanError {
  match self {
    Product =>
      numeric_expression(
        [BFloat16, BFloat16],
        Binary(
          MultiplyRN,
          Convert(Float32, Input(0)),
          Convert(Float32, Input(1)),
        ),
      )
    AddProduct =>
      numeric_expression([Float32, Float32], Binary(AddRN, Input(0), Input(1)))
    Publish => numeric_expression([Float32], Convert(BFloat16, Input(0)))
  }
}

///|
pub fn ScaledDotReductionLaw::publication_expression(
  self : Self,
) -> NumericExpression raise PrecisionPlanError {
  guard self is UnscaledBlockMatrixFloat32(block_columns=_, publication~) else {
    raise IllegalPlacement
  }
  numeric_expression(
    [Float32, Float32, Float32],
    match publication {
      SeparateMultiplyAdd =>
        Binary(AddRN, Input(0), Binary(MultiplyRN, Input(1), Input(2)))
      FusedMultiplyAdd => FusedMultiplyAdd(Input(1), Input(2), Input(0))
    },
  )
}
