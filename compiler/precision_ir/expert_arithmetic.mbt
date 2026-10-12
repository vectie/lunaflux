///|
pub(all) enum ExpertExpression {
  ClampGate
  ClampUp
  Product
  ApplyRoute
  DownPublish
  CombineUpdate
  DotUpdate
}

///|
pub fn ExpertMlpPrecision::expression(
  self : Self,
  operation : ExpertExpression,
) -> NumericExpression raise PrecisionPlanError {
  let x = Input(0)
  let y = Input(1)
  let z = Input(2)
  match operation {
    ClampGate | ClampUp => {
      let limited = Binary(MinimumNumber, Convert(Float32, x), y)
      numeric_expression(
        [BFloat16, Float32],
        match self.clamp {
          None => x
          Some(_) =>
            Convert(
              BFloat16,
              match operation {
                ClampGate => limited
                _ => Binary(MaximumNumber, Unary(Negate, y), limited)
              },
            )
        },
      )
    }
    Product => {
      let activated = Binary(
        DivideRN,
        x,
        Binary(
          AddRN,
          Float32Bits(0x3f800000U),
          Unary(NaturalExp, Unary(Negate, x)),
        ),
      )
      let activated = match self.activation_rounding {
        Bf16Stages => Convert(Float32, Convert(BFloat16, activated))
        F32Product => activated
      }
      numeric_expression(
        [Float32, BFloat16],
        Binary(MultiplyRN, activated, Convert(Float32, y)),
      )
    }
    ApplyRoute =>
      numeric_expression(
        [Float32, Float32],
        match self.routing_placement {
          BeforeDownInputRound => Binary(MultiplyRN, x, y)
          AfterDownOutputRound => x
        },
      )
    DownPublish =>
      numeric_expression(
        [Float32, Float32],
        match self.routing_placement {
          BeforeDownInputRound => Convert(BFloat16, x)
          AfterDownOutputRound =>
            Convert(
              BFloat16,
              Binary(MultiplyRN, Convert(Float32, Convert(BFloat16, x)), y),
            )
        },
      )
    CombineUpdate =>
      numeric_expression(
        [Float32, BFloat16],
        Binary(AddRN, x, Convert(Float32, y)),
      )
    DotUpdate =>
      numeric_expression(
        [Float32, Float32, Float32],
        Binary(AddRN, x, Binary(MultiplyRN, y, z)),
      )
  }
}
