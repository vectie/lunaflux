///|
pub(all) enum Slot {
  Counts
  Input
  Output
  RouterWeight
  RoutedBank
  SharedGateWeight
  SharedUpWeight
  SharedDownWeight
  SharedGateProjectionWeight
  RouterLogits
  SelectedIndices
  SelectedWeights
  ExpertMap
  ExpertActivation
  ExpertWeighted
  RoutedOutput
  SharedProduct
  SharedOutput
  SharedGateLogits
} derive(Debug, Eq)

///|
pub extend Slot with Eq::{equal, not_equal}

///|
pub extend Slot with @debug.Debug::{to_repr}

///|
pub fn Slot::name(self : Self) -> String {
  match self {
    Counts => "counts"
    Input => "input"
    Output => "output"
    RouterWeight => "router_weight"
    RoutedBank => "routed_bank"
    SharedGateWeight => "shared_gate_weight"
    SharedUpWeight => "shared_up_weight"
    SharedDownWeight => "shared_down_weight"
    SharedGateProjectionWeight => "shared_gate_projection_weight"
    RouterLogits => "router_logits"
    SelectedIndices => "selected_indices"
    SelectedWeights => "selected_weights"
    ExpertMap => "expert_map"
    ExpertActivation => "expert_activation"
    ExpertWeighted => "expert_weighted"
    RoutedOutput => "routed_output"
    SharedProduct => "shared_product"
    SharedOutput => "shared_output"
    SharedGateLogits => "shared_gate_logits"
  }
}

///|
/// None denotes a borrowed allocation; Some denotes a view into one reusable
/// workspace allocation. Offsets are aligned to 16 bytes, without aliasing.
pub(all) struct Region {
  slot : Slot
  workspace_offset : Int64?
  byte_count : Int64
  alignment : Int64
}

///|
fn regions(
  operators : @operators.DecoderOperators,
) -> (ReadOnlyArray[Region], Int64) raise {
  let plan = operators.routed_experts()
  let (rows, hidden, _, experts, _) = plan.geometry()
  let cells = rows.to_int64() * hidden.to_int64()
  let (_, expert_count, _) = operators.router_projection().geometry()
  let shared_weight_bytes = operators.shared_mlp().weight_bytes()
  let borrowed : Array[(Slot, Int64, Int64)] = [
    (Counts, 20L, 4L),
    (Input, cells * 2L, 2L),
    (Output, cells * 2L, 2L),
    (RouterWeight, expert_count.to_int64() * hidden.to_int64() * 2L, 2L),
    (RoutedBank, plan.bank_bytes(), 16L),
    (SharedGateWeight, shared_weight_bytes, 2L),
    (SharedUpWeight, shared_weight_bytes, 2L),
    (SharedDownWeight, shared_weight_bytes, 2L),
    (SharedGateProjectionWeight, hidden.to_int64() * 2L, 2L),
  ]
  let scratch : Array[(Slot, Int64, Int64)] = [
    (RouterLogits, rows.to_int64() * expert_count.to_int64() * 2L, 2L),
    (SelectedIndices, operators.routing().index_bytes(), 4L),
    (SelectedWeights, operators.routing().weight_bytes(), 4L),
    (ExpertMap, experts.to_int64() * 4L, 4L),
    (ExpertActivation, plan.workspace_bytes().0, 2L),
    (ExpertWeighted, plan.workspace_bytes().1, 2L),
    (RoutedOutput, cells * 4L, 4L),
    (SharedProduct, operators.shared_mlp().product_bytes(), 2L),
    (SharedOutput, cells * 2L, 2L),
    (SharedGateLogits, rows.to_int64() * 2L, 2L),
  ]
  let out : Array[Region] = borrowed.map(entry => {
    let (slot, bytes, alignment) = entry
    { slot, workspace_offset: None, byte_count: bytes, alignment, }
  })
  let mut offset = 0L
  for (slot, bytes, alignment) in scratch {
    guard offset <= 9223372036854775792L &&
      bytes <= 9223372036854775807L - (offset + 15L) / 16L * 16L else {
      raise @precision.PrecisionPlanError::ArithmeticOverflow
    }
    offset = (offset + 15L) / 16L * 16L
    out.push({
      slot,
      workspace_offset: Some(offset),
      byte_count: bytes,
      alignment,
    })
    offset += bytes
  }
  (ReadOnlyArray::from_array(out), offset)
}
