///|
pub struct Stage {
  priv symbol : String
  priv launch : (Int32, Int32, Int32)
  priv arguments : ReadOnlyArray[Slot]
}

///|
pub fn Stage::symbol(self : Self) -> String {
  self.symbol
}

///|
pub fn Stage::launch(self : Self) -> (Int32, Int32, Int32) {
  self.launch
}

///|
pub fn Stage::arguments(self : Self) -> ReadOnlyArray[Slot] {
  self.arguments
}

///|
/// Immutable startup program for one selected-model MoE. It owns source,
/// ordered ABIs and workspace extents, but no device resources or token state.
pub struct Program {
  priv operators : @operators.DecoderOperators
  priv source : Bytes
  priv stages : ReadOnlyArray[Stage]
  priv regions : ReadOnlyArray[Region]
  priv workspace_bytes : Int64
}

///|
pub fn Program::new(operators : @operators.DecoderOperators) -> Self raise {
  let routed = @expert.packed_expert_mlp_source(
    operators.routed_experts(),
    symbol="lunaflux_minicpm_v47_experts",
  )
  let dense = @dense.dense_mlp_source(
    operators.shared_mlp(),
    symbol_prefix="lunaflux_minicpm_v47_shared",
  )
  let router = @router.softmax_topk_source(
    operators.routing(),
    symbol="lunaflux_minicpm_v47_router",
  )
  let combine = @shared.shared_expert_source(
    operators.shared_combine(),
    symbol="lunaflux_minicpm_v47_shared_combine",
  )
  let router_symbol = "lunaflux_minicpm_v47_router_projection"
  let gate_symbol = "lunaflux_minicpm_v47_shared_gate_projection"
  let router_projection = operators.router_projection()
  let gate_projection = operators.shared_gate_projection()
  let stage = fn(symbol, launch, arguments) {
    { symbol, launch, arguments: ReadOnlyArray::from_array(arguments), }
  }
  let stages = [
    stage(router_symbol, (router_projection.tile_count(), 1, 128), [
      Counts,
      Input,
      RouterWeight,
      RouterLogits,
    ]),
    stage(router.symbol(), router.launch(), [
      Counts,
      RouterLogits,
      SelectedIndices,
      SelectedWeights,
    ]),
  ]
  let expert_arguments = [
    [Counts, Input, SelectedIndices, ExpertMap, RoutedBank, ExpertActivation],
    [
      Counts,
      ExpertActivation,
      SelectedIndices,
      SelectedWeights,
      ExpertMap,
      RoutedBank,
      ExpertWeighted,
    ],
    [Counts, SelectedIndices, ExpertWeighted, RoutedOutput],
  ]
  guard routed.symbols().length() == 3 && dense.execution_order().length() == 2 else {
    fail("unsupported MiniCPM MoE stage layout")
  }
  for i, symbol in routed.symbols() {
    let launch = routed.launches()[i]
    stages.push(stage(symbol, (launch.0, 1, launch.1), expert_arguments[i]))
  }
  let shared_arguments = [
    [Counts, Input, SharedGateWeight, SharedUpWeight, SharedProduct],
    [Counts, SharedProduct, SharedDownWeight, SharedOutput],
  ]
  for i in dense.execution_order() {
    let launch = dense.launch(i)
    stages.push(
      stage(dense.symbol(i), (launch.0, 1, launch.1), shared_arguments[i]),
    )
  }
  stages.push(
    stage(gate_symbol, (gate_projection.tile_count(), 1, 128), [
      Counts,
      Input,
      SharedGateProjectionWeight,
      SharedGateLogits,
    ]),
  )
  stages.push(
    stage(combine.symbol(), combine.launch(), [
      Counts,
      RoutedOutput,
      SharedOutput,
      SharedGateLogits,
      Output,
    ]),
  )
  let (regions, workspace_bytes) = regions(operators)
  {
    operators,
    source: @matrix.dense_source(router_projection, symbol=router_symbol) +
    b"\n" +
    router.source_bytes() +
    b"\n" +
    routed.source_bytes() +
    b"\n" +
    dense.source_bytes() +
    b"\n" +
    @matrix.dense_source(gate_projection, symbol=gate_symbol) +
    b"\n" +
    combine.source_bytes(),
    stages: ReadOnlyArray::from_array(stages),
    regions,
    workspace_bytes,
  }
}

///|
pub fn Program::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn Program::stages(self : Self) -> ReadOnlyArray[Stage] {
  self.stages
}

///|
pub fn Program::regions(self : Self) -> ReadOnlyArray[Region] {
  self.regions
}

///|
pub fn Program::workspace_bytes(self : Self) -> Int64 {
  self.workspace_bytes
}

///|
pub fn Program::region(self : Self, slot : Slot) -> Region {
  for region in self.regions {
    if region.slot == slot {
      return region
    }
  }
  abort("internal MiniCPM MoE slot")
}
