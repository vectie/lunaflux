///|
pub fn ExpertMlpPrecision::down_publication(
  self : Self,
) -> ExpertDownPublication {
  self.down_publication
}

///|
/// Pure intermediate-axis partition of every local expert. Expert identity
/// ownership does not change (this is TP, not EP). Each rank computes all
/// routed jobs over disjoint intermediate channels.
pub struct ExpertTensorPartition {
  priv rank : Int
  priv world_size : Int
  priv intermediate_start : Int
  priv collective_width : Int
  priv slices : ReadOnlyArray[PackedMatrixSlice]
  priv rank_plan : ExpertMlpPrecision
}

///|
pub fn ExpertTensorPartition::new(
  original : ExpertMlpPrecision,
  rank~ : Int,
  world_size~ : Int,
  memory_ceiling~ : Int64,
) -> Self raise PrecisionPlanError {
  let (rows, hidden, intermediate, experts, selected) = original.geometry()
  guard world_size > 1 &&
    rank >= 0 &&
    rank < world_size &&
    intermediate % world_size == 0 &&
    original.down_publication() == LocalBFloat16 &&
    original.local_experts().length() == experts &&
    original.activation_quantization() is None &&
    original.scaled_dot_law() == DecodeOperandsFloat32 else {
    raise NoImplementation
  }
  let width = intermediate / world_size
  let cells = hidden.to_int64() * selected.to_int64()
  guard cells <= 2147483647L else { raise ArithmeticOverflow }
  let start = rank * width
  let slices = []
  let operands = []
  for index, weight in original.weights() {
    let view = PackedMatrixSlice::new(
      weight.representation(),
      row_start=if index < 2 { start } else { 0 },
      rows=if index < 2 { width } else { hidden },
      column_start=if index < 2 { 0 } else { start },
      columns=if index < 2 { hidden } else { width },
    )
    slices.push(view)
    operands.push(
      plan_operand(
        role=Weight,
        representation=view.representation(),
        placement=DecodeInOperandTile,
        supported_placements=[DecodeInOperandTile],
        accuracy=weight.accuracy(),
        memory_ceiling~,
        tile_elements=32,
      ),
    )
  }
  let rank_plan = ExpertMlpPrecision::new(
    gate=operands[0],
    up=operands[1],
    down=operands[2],
    expert_count=experts,
    local_experts=original.local_experts(),
    selected_experts=selected,
    maximum_rows=rows,
    clamp?=original.clamp(),
    activation_rounding=original.activation_rounding(),
    routing_placement=original.routing_placement(),
    dot_law=original.dot_law(),
    down_publication=ReduceFloat32BeforeRound,
    memory_ceiling~,
  )
  {
    rank,
    world_size,
    intermediate_start: start,
    collective_width: cells.to_int(),
    slices: ReadOnlyArray::from_array(slices),
    rank_plan,
  }
}

///|
pub fn ExpertTensorPartition::rank(self : Self) -> Int {
  self.rank
}

///|
pub fn ExpertTensorPartition::world_size(self : Self) -> Int {
  self.world_size
}

///|
pub fn ExpertTensorPartition::intermediate_start(self : Self) -> Int {
  self.intermediate_start
}

///|
pub fn ExpertTensorPartition::slices(
  self : Self,
) -> ReadOnlyArray[PackedMatrixSlice] {
  self.slices
}

///|
pub fn ExpertTensorPartition::rank_plan(self : Self) -> ExpertMlpPrecision {
  self.rank_plan
}

///|
/// F32 all-reduce width per live token, before down publication. Routing order
/// must be replicated across ranks; no rounded/shared output enters this sum.
pub fn ExpertTensorPartition::collective_width(self : Self) -> Int {
  self.collective_width
}
