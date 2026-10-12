///|
pub(all) enum HyperMixExpression {
  BranchProduct
  ResidualUpdate
  Publish
}

///|
pub fn HyperConnectionMixLaw::expression(
  self : Self,
  operation : HyperMixExpression,
) -> NumericExpression raise PrecisionPlanError {
  let x = Input(0)
  let y = Input(1)
  let z = Input(2)
  fn rounded(x : NumericExpr) -> NumericExpr {
    Convert(Float32, Convert(BFloat16, x))
  }
  match operation {
    BranchProduct =>
      numeric_expression(
        [Float32, Float32],
        match self {
          RoundedBf16ControlsAndProducts =>
            rounded(Binary(MultiplyRN, rounded(x), y))
          _ => Binary(MultiplyRN, x, y)
        },
      )
    ResidualUpdate =>
      numeric_expression(
        [Float32, Float32, Float32],
        Binary(
          AddRN,
          x,
          Binary(
            MultiplyRN,
            match self {
              RoundedBf16ControlsAndProducts => rounded(y)
              _ => y
            },
            z,
          ),
        ),
      )
    Publish =>
      numeric_expression(
        [Float32, Float32],
        Convert(
          BFloat16,
          match self {
            OrderedF32SingleBf16Round => x // branch was the residual fold's seed
            TransposedF32ResidualThenBranch => Binary(AddRN, x, y)
            RoundedBf16ControlsAndProducts => Binary(AddRN, x, rounded(y))
          },
        ),
      )
  }
}
