///|
pub(all) enum HyperControlExpression {
  NormalizeProduct
  PreOffset
  PostScale
  SoftmaxWeight
  SoftmaxPublish
  CollapseUpdate
  Publish
  SinkhornNormalize
}

///|
pub fn HyperControlExpression::expression(
  self : Self,
) -> NumericExpression raise PrecisionPlanError {
  let x = Input(0)
  let y = Input(1)
  let z = Input(2)
  match self {
    NormalizeProduct =>
      numeric_expression([Float32, Float32], Binary(MultiplyRN, x, y))
    PreOffset => numeric_expression([Float32, Float32], Binary(AddRN, x, y))
    PostScale =>
      numeric_expression(
        [Float32],
        Binary(MultiplyRN, Float32Bits(0x40000000U), x),
      )
    SoftmaxWeight =>
      numeric_expression(
        [Float32, Float32],
        Unary(NaturalExp, Binary(AddRN, x, Unary(Negate, y))),
      )
    SoftmaxPublish =>
      numeric_expression(
        [Float32, Float32, Float32],
        Binary(AddRN, Binary(DivideRN, x, y), z),
      )
    CollapseUpdate =>
      numeric_expression(
        [Float32, Float32, BFloat16],
        Binary(AddRN, x, Binary(MultiplyRN, y, Convert(Float32, z))),
      )
    Publish => numeric_expression([Float32], Convert(BFloat16, x))
    SinkhornNormalize =>
      numeric_expression(
        [Float32, Float32, Float32],
        Binary(DivideRN, x, Binary(AddRN, y, z)),
      )
  }
}
