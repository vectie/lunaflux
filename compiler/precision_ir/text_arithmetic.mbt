///|
/// Numerical boundaries of mean-stream text egress. Operands are scalars from
/// the enclosing read/effect plan. Rounding, including intermediate BF16,
/// remains explicit; a higher-precision variant is not an exact rewrite.
pub(all) enum TextExpression {
  StreamUpdate
  Mean
  SquareUpdate
  Inverse
  Normalize
  Weight
  HeadUpdate
  Logits
}

///|
pub fn TextIoPrecision::expression(
  self : Self,
  operation : TextExpression,
) -> NumericExpression raise PrecisionPlanError {
  let x : NumericExpr = Input(0)
  let y : NumericExpr = Input(1)
  let z : NumericExpr = Input(2)
  let (types, root) : (Array[NumericScalar], NumericExpr) = match operation {
    StreamUpdate => ([Float32, BFloat16], Binary(AddRN, x, Convert(Float32, y)))
    Mean =>
      (
        [Float32, Float32],
        Convert(Float32, Convert(BFloat16, Binary(DivideRN, x, y))),
      )
    SquareUpdate =>
      ([Float32, Float32], Binary(AddRN, x, Binary(MultiplyRN, y, y)))
    Inverse =>
      (
        [Float32, Float32, Float32],
        Unary(ReciprocalSqrt, Binary(AddRN, Binary(DivideRN, x, y), z)),
      )
    Normalize => {
      let product = Binary(MultiplyRN, x, y)
      (
        [Float32, Float32],
        match self.normalization_law {
          Bf16NormalizeThenWeight =>
            Convert(Float32, Convert(BFloat16, product))
          F32NormalizeThenWeight => product
        },
      )
    }
    Weight =>
      (
        [Float32, BFloat16],
        Convert(BFloat16, Binary(MultiplyRN, x, Convert(Float32, y))),
      )
    HeadUpdate =>
      (
        [Float32, BFloat16, BFloat16],
        Binary(
          AddRN,
          x,
          Binary(MultiplyRN, Convert(Float32, y), Convert(Float32, z)),
        ),
      )
    Logits =>
      (
        [Float32],
        match self.logits_law {
          Bf16Round => Convert(Float32, Convert(BFloat16, x))
          F32Accumulate => x
        },
      )
  }
  NumericExpression::new(types.map(kind => (kind, AnyFloat)), root)
}

///|
pub fn TextIoPrecision::fold(
  self : Self,
  operation : TextExpression,
) -> OrderedNumericFold raise PrecisionPlanError {
  guard !(operation is HeadUpdate) || self.head_law() == ScalarFloat32 else {
    raise NoImplementation
  }
  OrderedNumericFold::new(
    extent=match operation {
      StreamUpdate => self.streams
      SquareUpdate | HeadUpdate => self.hidden
      _ => raise IllegalPlacement
    },
    initial_bits=0U,
    update=self.expression(operation),
  )
}
