///|
pub(all) enum SoftmaxTopKExpression {
  ShiftedExponential
  Sum
  Probability
  SelectedWeight
}

///|
/// Both alternatives preserve a BF16 value round. F32 storage is an exact
/// widening for consumers whose existing route-weight ABI takes float.
pub(all) enum RoutingWeightStorage {
  Bf16Weights
  Float32CarryingBf16
}

///|
/// BF16 logits -> FP32 full softmax -> descending top K -> FP32 selected
/// renormalization -> BF16 weights. Equal probabilities prefer lower IDs.
/// Reduction order is ascending expert ID, then descending selected rank.
pub struct SoftmaxTopKPrecision {
  priv experts : Int32
  priv selected : Int32
  priv rows : Int32
  priv weight_storage : RoutingWeightStorage
}

///|
pub fn SoftmaxTopKPrecision::new(
  expert_count~ : Int32,
  experts_per_token~ : Int32,
  maximum_rows~ : Int32,
  weight_storage~ : RoutingWeightStorage = Bf16Weights,
) -> SoftmaxTopKPrecision raise PrecisionPlanError {
  guard expert_count > 0 &&
    experts_per_token > 0 &&
    experts_per_token <= expert_count &&
    maximum_rows > 0 else {
    raise IllegalPlacement
  }
  ignore(
    expert_checked_mul(maximum_rows.to_int64(), expert_count.to_int64() * 2L),
  )
  ignore(
    expert_checked_mul(
      maximum_rows.to_int64(),
      experts_per_token.to_int64() * 4L,
    ),
  )
  {
    experts: expert_count,
    selected: experts_per_token,
    rows: maximum_rows,
    weight_storage,
  }
}

///|
pub fn SoftmaxTopKPrecision::weight_storage(
  self : Self,
) -> RoutingWeightStorage {
  self.weight_storage
}

///|
pub fn SoftmaxTopKPrecision::geometry(
  self : SoftmaxTopKPrecision,
) -> (Int32, Int32, Int32) {
  (self.experts, self.selected, self.rows)
}

///|
pub fn SoftmaxTopKPrecision::weight_bytes(self : SoftmaxTopKPrecision) -> Int64 {
  self.rows.to_int64() *
  self.selected.to_int64() *
  (if self.weight_storage is Bf16Weights { 2L } else { 4L })
}

///|
pub fn SoftmaxTopKPrecision::index_bytes(self : SoftmaxTopKPrecision) -> Int64 {
  self.rows.to_int64() * self.selected.to_int64() * 4L
}

///|
pub fn SoftmaxTopKPrecision::expression(
  self : SoftmaxTopKPrecision,
  expression : SoftmaxTopKExpression,
) -> NumericExpression raise PrecisionPlanError {
  let x = Input(0)
  let y = Input(1)
  match expression {
    ShiftedExponential =>
      numeric_expression(
        [BFloat16, Float32],
        Unary(NaturalExp, Binary(SubtractRN, Convert(Float32, x), y)),
      )
    Sum => numeric_expression([Float32, Float32], Binary(AddRN, x, y))
    Probability =>
      numeric_expression([Float32, Float32], Binary(DivideRN, x, y))
    SelectedWeight => {
      let rounded = Convert(BFloat16, Binary(DivideRN, x, y))
      numeric_expression(
        [Float32, Float32],
        if self.weight_storage is Bf16Weights {
          rounded
        } else {
          Convert(Float32, rounded)
        },
      )
    }
  }
}
