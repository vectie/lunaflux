///|
/// Threshold is bound to the validated plan's F32 literal by lowering. Both
/// arms retain strict exp/log1p, and selection does not evaluate the unused arm.
pub fn MoeRoutingPrecision::score_expression(
  self : Self,
) -> NumericExpression raise PrecisionPlanError {
  let x = Input(0)
  let softplus = match self.softplus_law {
    StableFloat32 =>
      Select(
        GreaterThan,
        x,
        Float32Bits(0U),
        Binary(
          AddRN,
          x,
          Unary(NaturalLogOnePlus, Unary(NaturalExp, Unary(Negate, x))),
        ),
        Unary(NaturalLogOnePlus, Unary(NaturalExp, x)),
      )
    ThresholdedFloat32(_) =>
      Select(
        GreaterThan,
        x,
        Input(1),
        x,
        Unary(NaturalLogOnePlus, Unary(NaturalExp, x)),
      )
  }
  numeric_expression(
    [Float32, Float32],
    match self.score {
      Sigmoid =>
        Binary(
          DivideRN,
          Float32Bits(0x3f800000U),
          Binary(
            AddRN,
            Float32Bits(0x3f800000U),
            Unary(NaturalExp, Unary(Negate, x)),
          ),
        )
      SqrtSoftplus => Unary(SquareRoot, softplus)
    },
  )
}

///|
pub fn MoeCombinePrecision::expression(
  self : Self,
) -> NumericExpression raise PrecisionPlanError {
  let routed = match self.rounding {
    SumThenBf16 => Input(0)
    Bf16RoutedThenSum => Convert(Float32, Convert(BFloat16, Input(0)))
  }
  numeric_expression(
    [Float32, Float32],
    Convert(BFloat16, Binary(AddRN, routed, Input(1))),
  )
}

///|
/// Arithmetic for selected weights. Selection/tie-breaking and reduction
/// topology remain separate existing semantic plans, never implicit sorting.
pub(all) enum RoutingWeightExpression {
  SumUpdate
  Denominator
  Publish
  ScaleOnly
}

///|
pub fn RoutingWeightExpression::program(
  self : Self,
) -> NumericExpression raise PrecisionPlanError {
  match self {
    SumUpdate | Denominator =>
      numeric_expression([Float32, Float32], Binary(AddRN, Input(0), Input(1)))
    Publish =>
      numeric_expression(
        [Float32, Float32, Float32],
        Binary(MultiplyRN, Binary(DivideRN, Input(0), Input(1)), Input(2)),
      )
    ScaleOnly =>
      numeric_expression(
        [Float32, Float32],
        Binary(MultiplyRN, Input(0), Input(1)),
      )
  }
}
