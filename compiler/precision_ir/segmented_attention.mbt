///|
pub(all) enum SegmentedAttentionExpression {
  DotUpdate
  ScaledScore
  ProbabilityStorage
  ValueUpdate
  Publish
}

///|
/// Full noncausal attention within supplied packed row spans. BF16 Q/K/V,
/// BF16 dot output, BF16 scaled scores, F32 global softmax, BF16 normalized
/// probabilities, F32 value accumulation and BF16 output. No rotary or KV cache.
pub struct SegmentedAttentionPrecision {
  priv rows : Int32
  priv segment : Int32
  priv heads : Int32
  priv width : Int32
}

///|
pub fn SegmentedAttentionPrecision::new(
  maximum_rows~ : Int32,
  maximum_segment~ : Int32,
  head_count~ : Int32,
  head_width~ : Int32,
) -> Self raise PrecisionPlanError {
  guard maximum_rows > 0 &&
    maximum_segment > 0 &&
    maximum_segment <= maximum_rows &&
    head_count > 0 &&
    head_width > 0 else {
    raise IllegalPlacement
  }
  let elements = expert_checked_mul(
    expert_checked_mul(maximum_rows.to_int64(), head_count.to_int64()),
    head_width.to_int64(),
  )
  guard elements <= 2147483647L else { raise ArithmeticOverflow }
  {
    rows: maximum_rows,
    segment: maximum_segment,
    heads: head_count,
    width: head_width,
  }
}

///|
pub fn SegmentedAttentionPrecision::geometry(
  self : Self,
) -> (Int32, Int32, Int32, Int32) {
  (self.rows, self.segment, self.heads, self.width)
}

///|
pub fn SegmentedAttentionPrecision::expression(
  _self : Self,
  operation : SegmentedAttentionExpression,
) -> NumericExpression raise PrecisionPlanError {
  let x = Input(0)
  let y = Input(1)
  let z = Input(2)
  match operation {
    DotUpdate =>
      numeric_expression(
        [Float32, BFloat16, BFloat16],
        FusedMultiplyAdd(Convert(Float32, y), Convert(Float32, z), x),
      )
    ValueUpdate =>
      numeric_expression(
        [Float32, Float32, BFloat16],
        FusedMultiplyAdd(y, Convert(Float32, z), x),
      )
    ScaledScore =>
      numeric_expression(
        [Float32, Float32],
        Convert(
          Float32,
          Convert(
            BFloat16,
            Binary(MultiplyRN, Convert(Float32, Convert(BFloat16, x)), y),
          ),
        ),
      )
    ProbabilityStorage =>
      numeric_expression(
        [Float32, Float32],
        Convert(Float32, Convert(BFloat16, Binary(DivideRN, x, y))),
      )
    Publish => numeric_expression([Float32], Convert(BFloat16, x))
  }
}
