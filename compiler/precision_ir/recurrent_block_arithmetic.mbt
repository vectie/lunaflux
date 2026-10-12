///|
pub(all) enum RecurrentBlockExpression {
  Beta
  OutputGate
  BiasedForget
  DecayRate
  ScaleForget
  BoundDecay
}

///|
pub fn RecurrentBlockExpression::program(
  self : Self,
) -> NumericExpression raise PrecisionPlanError {
  let x = Input(0)
  let y = Input(1)
  let z = Input(2)
  let w = Input(3)
  fn sigmoid(x : NumericExpr) -> NumericExpr {
    Binary(
      DivideRN,
      Float32Bits(0x3f800000U),
      Binary(
        AddRN,
        Float32Bits(0x3f800000U),
        Unary(NaturalExp, Unary(Negate, x)),
      ),
    )
  }
  match self {
    Beta =>
      numeric_expression(
        [BFloat16],
        Convert(BFloat16, sigmoid(Convert(Float32, x))),
      )
    OutputGate =>
      numeric_expression(
        [BFloat16, Float32, BFloat16, BFloat16],
        Convert(
          BFloat16,
          Binary(
            MultiplyRN,
            Binary(
              MultiplyRN,
              Convert(Float32, z),
              Binary(MultiplyRN, Convert(Float32, x), y),
            ),
            sigmoid(Convert(Float32, w)),
          ),
        ),
      )
    BiasedForget =>
      numeric_expression(
        [BFloat16, Float32],
        Binary(AddRN, Convert(Float32, x), y),
      )
    DecayRate => numeric_expression([Float32], Unary(NaturalExp, x))
    ScaleForget | BoundDecay =>
      numeric_expression([Float32, Float32], Binary(MultiplyRN, x, y))
  }
}
