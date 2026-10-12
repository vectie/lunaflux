///|
fn numeric_expression(
  kinds : ArrayView[NumericScalar],
  root : NumericExpr,
) -> NumericExpression raise PrecisionPlanError {
  NumericExpression::new(kinds.map(kind => (kind, AnyFloat)), root)
}

///|
/// Accumulator is the third operand. Contraction is a numerical choice,
/// independent of the physical reduction topology and device instruction set.
pub fn Float32AffineLaw::expression(
  self : Self,
) -> NumericExpression raise PrecisionPlanError {
  numeric_expression(
    [Float32, Float32, Float32],
    match self {
      SeparateMultiplyAdd =>
        Binary(AddRN, Binary(MultiplyRN, Input(0), Input(1)), Input(2))
      FusedMultiplyAdd => FusedMultiplyAdd(Input(0), Input(1), Input(2))
    },
  )
}

///|
pub fn SigmoidEvaluationLaw::expression(
  self : Self,
) -> NumericExpression raise PrecisionPlanError {
  let one = Float32Bits(0x3f800000U)
  let x = Input(0)
  let direct = Binary(
    DivideRN,
    one,
    Binary(AddRN, one, Unary(NaturalExp, Unary(Negate, x))),
  )
  numeric_expression(
    [Float32],
    match self {
      DirectReciprocal => direct
      StableBranches =>
        Select(
          GreaterOrEqual,
          x,
          Float32Bits(0U),
          direct,
          Binary(
            DivideRN,
            Unary(NaturalExp, x),
            Binary(AddRN, one, Unary(NaturalExp, x)),
          ),
        )
    },
  )
}

///|
pub fn ReciprocalRootLaw::expression(
  self : Self,
) -> NumericExpression raise PrecisionPlanError {
  numeric_expression(
    [Float32],
    match self {
      ReciprocalSqrtFloat32 => Unary(ReciprocalSqrt, Input(0))
      DivideRoundedSqrtFloat32 =>
        Binary(DivideRN, Float32Bits(0x3f800000U), Unary(SquareRoot, Input(0)))
    },
  )
}

///|
pub fn RotaryProductLaw::expression(
  self : Self,
  imaginary~ : Bool,
) -> NumericExpression raise PrecisionPlanError {
  let l = Input(0)
  let r = Input(1)
  let c = Input(2)
  let s = Input(3)
  let result = match (self, imaginary) {
    (SeparateFloat32Products, false) =>
      Binary(
        AddRN,
        Binary(MultiplyRN, l, c),
        Unary(Negate, Binary(MultiplyRN, r, s)),
      )
    (SeparateFloat32Products, true) =>
      Binary(AddRN, Binary(MultiplyRN, l, s), Binary(MultiplyRN, r, c))
    (ComplexFloat32Fma, false) =>
      FusedMultiplyAdd(l, c, Unary(Negate, Binary(MultiplyRN, r, s)))
    (ComplexFloat32Fma, true) =>
      FusedMultiplyAdd(r, c, Binary(MultiplyRN, l, s))
  }
  numeric_expression(
    [Float32, Float32, Float32, Float32],
    Convert(BFloat16, result),
  )
}
