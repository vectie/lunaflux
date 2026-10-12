///|
pub(all) enum RecurrentDeltaExpression {
  SquareUpdate
  NormInverse
  QueryScale
  QueryNormalize
  KeyNormalize
  Decay
  StateDecay
  DotUpdate
  Delta
  StateUpdate
  Publish
}

///|
pub fn RecurrentDeltaPrecision::expression(
  self : Self,
  operation : RecurrentDeltaExpression,
) -> NumericExpression raise PrecisionPlanError {
  ignore(self)
  let x = Input(0)
  let y = Input(1)
  let z = Input(2)
  match operation {
    SquareUpdate =>
      numeric_expression(
        [Float32, Float32],
        Binary(AddRN, x, Binary(MultiplyRN, y, y)),
      )
    NormInverse =>
      numeric_expression(
        [Float32, Float32],
        Binary(
          DivideRN,
          Float32Bits(0x3f800000U),
          Unary(SquareRoot, Binary(AddRN, x, y)),
        ),
      )
    QueryScale =>
      numeric_expression(
        [Float32],
        Binary(DivideRN, Float32Bits(0x3f800000U), Unary(SquareRoot, x)),
      )
    QueryNormalize =>
      numeric_expression(
        [Float32, Float32, Float32],
        Binary(MultiplyRN, Binary(MultiplyRN, x, y), z),
      )
    KeyNormalize | StateDecay =>
      numeric_expression([Float32, Float32], Binary(MultiplyRN, x, y))
    Decay => numeric_expression([Float32], Unary(NaturalExp, x))
    DotUpdate | StateUpdate =>
      numeric_expression(
        [Float32, Float32, Float32],
        Binary(AddRN, x, Binary(MultiplyRN, y, z)),
      )
    Delta =>
      numeric_expression(
        [BFloat16, Float32, Float32],
        Binary(MultiplyRN, Binary(SubtractRN, Convert(Float32, x), y), z),
      )
    Publish => numeric_expression([Float32], Convert(BFloat16, x))
  }
}

///|
pub fn RecurrentDeltaPrecision::fold(
  self : Self,
  operation : RecurrentDeltaExpression,
) -> OrderedNumericFold raise PrecisionPlanError {
  guard operation is (SquareUpdate | DotUpdate) else { raise IllegalPlacement }
  OrderedNumericFold::new(
    extent=self.keys,
    initial_bits=0U,
    update=self.expression(operation),
  )
}

///|
pub fn CausalConvolutionPrecision::projection_fold(
  self : Self,
) -> OrderedNumericFold raise PrecisionPlanError {
  OrderedNumericFold::new(
    extent=self.kernel,
    initial_bits=0U,
    update=numeric_expression(
      [Float32, BFloat16, BFloat16],
      Binary(
        AddRN,
        Input(0),
        Binary(
          MultiplyRN,
          Convert(Float32, Input(1)),
          Convert(Float32, Input(2)),
        ),
      ),
    ),
  )
}

///|
pub fn CausalConvolutionPrecision::activation_expression(
  self : Self,
) -> NumericExpression raise PrecisionPlanError {
  ignore(self)
  numeric_expression(
    [Float32],
    Convert(
      BFloat16,
      Binary(
        DivideRN,
        Input(0),
        Binary(
          AddRN,
          Float32Bits(0x3f800000U),
          Unary(NaturalExp, Unary(Negate, Input(0))),
        ),
      ),
    ),
  )
}
