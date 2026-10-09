///|
/// Score arithmetic is a numeric contract, not a CUDA or model-family choice.
pub(all) enum MoeRoutingScore {
  Sigmoid
  SqrtSoftplus
} derive(Eq, @debug.Debug)

///|
pub extend MoeRoutingScore with Eq::{equal, not_equal}

///|
pub extend MoeRoutingScore with @debug.Debug::{to_repr}

///|
/// BF16 hidden rows, BF16/F32 router weights, F32 logits and selection scores.
/// Stable lower-ID tie selection is explicit; correction changes choice only.
pub struct MoeRoutingPrecision {
  priv weight : OperandPlan
  priv rows : Int
  priv groups : Int
  priv group_topk : Int
  priv selected_groups : Int
  priv selected_experts : Int
  priv score : MoeRoutingScore
  priv normalize : Bool
  priv scaling : Double
  priv buffers : ReadOnlyArray[Int64]
  priv workspace : Int64
}

///|
pub fn MoeRoutingPrecision::new(
  weight~ : OperandPlan,
  maximum_rows~ : Int,
  group_count~ : Int,
  group_score_topk~ : Int,
  selected_group_count~ : Int,
  selected_experts~ : Int,
  score~ : MoeRoutingScore,
  normalize~ : Bool,
  scaling~ : Double,
  maximum_workspace_bytes~ : Int64,
) -> Self raise PrecisionPlanError {
  let (experts, _) = weight.representation().shape()
  guard weight.role() == Weight &&
    weight.effect() == Pure &&
    weight.placement() is (NativeOperand | DecodeInOperandTile) &&
    weight.representation().format() is (Bf16 | F32) &&
    maximum_rows > 0 &&
    group_count > 0 &&
    experts % group_count == 0 &&
    group_score_topk > 0 &&
    group_score_topk <= experts / group_count &&
    selected_group_count > 0 &&
    selected_group_count <= group_count &&
    selected_experts > 0 &&
    selected_experts <= selected_group_count * (experts / group_count) else {
    raise IllegalPlacement
  }
  guard !scaling.is_nan() && !scaling.is_inf() && scaling > 0.0 else {
    raise MissingAccuracyContract
  }
  let logits = expert_checked_mul(
    expert_checked_mul(maximum_rows.to_int64(), experts.to_int64()),
    4L,
  )
  let selected = expert_checked_mul(
    expert_checked_mul(maximum_rows.to_int64(), selected_experts.to_int64()),
    4L,
  )
  let buffers = [logits, logits, logits, selected, selected]
  let workspace = for bytes in buffers; total = 0L {
    guard bytes <= maximum_workspace_bytes - total else {
      raise MemoryBudgetExceeded
    }
    continue total + bytes
  } nobreak {
    total
  }
  {
    weight,
    rows: maximum_rows,
    groups: group_count,
    group_topk: group_score_topk,
    selected_groups: selected_group_count,
    selected_experts,
    score,
    normalize,
    scaling,
    buffers: ReadOnlyArray::from_array(buffers),
    workspace,
  }
}

///|
pub fn MoeRoutingPrecision::weight(self : Self) -> OperandPlan {
  self.weight
}

///|
pub fn MoeRoutingPrecision::geometry(self : Self) -> (Int, Int, Int, Int) {
  let (experts, hidden) = self.weight.representation().shape()
  (self.rows, hidden, experts, self.selected_experts)
}

///|
pub fn MoeRoutingPrecision::grouping(self : Self) -> (Int, Int, Int) {
  (self.groups, self.group_topk, self.selected_groups)
}

///|
pub fn MoeRoutingPrecision::score(self : Self) -> MoeRoutingScore {
  self.score
}

///|
pub fn MoeRoutingPrecision::normalize(self : Self) -> Bool {
  self.normalize
}

///|
pub fn MoeRoutingPrecision::scaling(self : Self) -> Double {
  self.scaling
}

///|
/// Logits, unbiased scores, choice scores, indices, selected scores, in order.
pub fn MoeRoutingPrecision::buffer_bytes(self : Self) -> ReadOnlyArray[Int64] {
  self.buffers
}

///|
pub fn MoeRoutingPrecision::workspace_bytes(self : Self) -> Int64 {
  self.workspace
}
