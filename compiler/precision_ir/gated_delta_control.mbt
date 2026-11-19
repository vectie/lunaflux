///|
pub(all) enum GatedDeltaControlExpression {
  Beta
  LogDecay
}

///|
/// BF16 projected a/b and BF16 per-head A_log/dt_bias. Beta publishes BF16;
/// log decay stays FP32 with threshold-20 softplus and no clipping.
pub struct GatedDeltaControlPrecision {
  priv heads : Int32
  priv rows : Int32
}

///|
pub fn GatedDeltaControlPrecision::new(
  head_count~ : Int32,
  maximum_rows~ : Int32,
) -> GatedDeltaControlPrecision raise PrecisionPlanError {
  guard head_count > 0 && maximum_rows > 0 else { raise IllegalPlacement }
  ignore(
    expert_checked_mul(head_count.to_int64() * maximum_rows.to_int64(), 4L),
  )
  { heads: head_count, rows: maximum_rows, }
}

///|
pub fn GatedDeltaControlPrecision::geometry(
  self : GatedDeltaControlPrecision,
) -> (Int32, Int32) {
  (self.heads, self.rows)
}

///|
pub fn GatedDeltaControlPrecision::expression(
  _self : GatedDeltaControlPrecision,
  expression : GatedDeltaControlExpression,
) -> NumericExpression raise PrecisionPlanError {
  let one = Float32Bits(0x3f800000U)
  match expression {
    Beta =>
      numeric_expression(
        [BFloat16],
        Convert(
          BFloat16,
          Binary(
            DivideRN,
            one,
            Binary(
              AddRN,
              one,
              Unary(NaturalExp, Unary(Negate, Convert(Float32, Input(0)))),
            ),
          ),
        ),
      )
    LogDecay => {
      let a_bias = Binary(
        AddRN,
        Convert(Float32, Input(0)),
        Convert(Float32, Input(2)),
      )
      let softplus = Select(
        GreaterThan,
        a_bias,
        Float32Bits(0x41a00000U),
        a_bias,
        Unary(NaturalLogOnePlus, Unary(NaturalExp, a_bias)),
      )
      numeric_expression(
        [BFloat16, BFloat16, BFloat16],
        Binary(
          MultiplyRN,
          Unary(Negate, Unary(NaturalExp, Convert(Float32, Input(1)))),
          softplus,
        ),
      )
    }
  }
}
