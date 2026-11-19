///|
pub(all) enum SharedExpertExpression {
  Sigmoid
  Weight
  Combine
}

///|
/// BF16 shared MLP result and gate logit, FP32 routed result carrying BF16
/// each-add output. Sigmoid, shared product and final sum each round to BF16.
pub struct SharedExpertPrecision {
  priv width : Int32
  priv rows : Int32
}

///|
pub fn SharedExpertPrecision::new(
  hidden_width~ : Int32,
  maximum_rows~ : Int32,
) -> Self raise PrecisionPlanError {
  guard hidden_width > 0 && maximum_rows > 0 else { raise IllegalPlacement }
  ignore(
    expert_checked_mul(hidden_width.to_int64() * maximum_rows.to_int64(), 4L),
  )
  { width: hidden_width, rows: maximum_rows, }
}

///|
pub fn SharedExpertPrecision::geometry(self : Self) -> (Int32, Int32) {
  (self.width, self.rows)
}

///|
pub fn SharedExpertPrecision::expression(
  _self : Self,
  expression : SharedExpertExpression,
) -> NumericExpression raise PrecisionPlanError {
  let x = Input(0)
  let y = Input(1)
  match expression {
    Sigmoid => Bf16GateExpression::Sigmoid.program()
    Weight => Bf16GateExpression::Product.program()
    Combine =>
      numeric_expression(
        [Float32, BFloat16],
        Convert(BFloat16, Binary(AddRN, x, Convert(Float32, y))),
      )
  }
}
