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
/// BF16 recurrent branch, excluding decoder input norm and residual. A single
/// immutable owner supplies sources, bindings and workspace to startup/runtime.
pub struct Program {
  priv source : Bytes
  priv stages : ReadOnlyArray[Stage]
  priv regions : ReadOnlyArray[Region]
  priv workspace_bytes : Int64
}

///|
pub fn Program::new(operators : @operators.DecoderOperators) -> Self raise {
  let delta = operators.recurrent()
  let conv = operators.convolution()
  let (heads, keys, values) = delta.geometry()
  let (rows, sequences, slots) = delta.capacity()
  let (channels, _) = conv.geometry()
  guard conv.capacity() == (rows, sequences, slots) &&
    channels == delta.query_key_head_count() * keys * 2 + heads * values &&
    delta.query_key_row_stride() == channels.to_int64() &&
    delta.value_row_stride() == channels.to_int64() else {
    fail("recurrent layout mismatch")
  }
  let make = fn(symbol, launch, arguments) {
    { symbol, launch, arguments: ReadOnlyArray::from_array(arguments), }
  }
  let stages : Array[Stage] = []
  let mut source = b""
  let projections : Array[(String, @operators.LinearProjection, Slot, Slot)] = [
    ("qkv", QueryKeyValue, QkvWeight, RawQkv),
    ("z", Gate, GateWeight, Gate),
    ("a", Decay, DecayWeight, Decay),
    ("b", Beta, BetaWeight, Beta),
  ]
  for (name, site, weight, output) in projections {
    let plan = operators.linear_projection(site)
    let symbol = "lunaflux_minicpm_v47_linear_" + name
    source += @matrix.dense_source(plan, symbol~) + b"\n"
    stages.push(
      make(symbol, (plan.tile_count(), 1, 128), [Counts, Input, weight, output]),
    )
  }
  let convolution = @conv.causal_convolution_source(
    conv,
    symbol="lunaflux_minicpm_v47_conv",
  )
  source += convolution.source_bytes() + b"\n"
  stages.push(
    make(convolution.symbol(), convolution.launch(), [
      Counts,
      RowOffsets,
      RequestSlots,
      Reset,
      RawQkv,
      ConvolutionWeight,
      ConvolutionState,
      ConvolvedQuery,
    ]),
  )
  let controls = @controls.gated_delta_control_source(
    operators.controls(),
    symbol="lunaflux_minicpm_v47_controls",
  )
  source += controls.source_bytes() + b"\n"
  stages.push(
    make(controls.symbol(), controls.launch(), [
      Counts,
      Decay,
      Beta,
      ALog,
      DtBias,
      LogDecay,
      BetaControl,
    ]),
  )
  let recurrence = @delta.recurrent_delta_source(
    delta,
    symbol="lunaflux_minicpm_v47_delta",
  )
  source += recurrence.source_bytes() + b"\n"
  stages.push(
    make(recurrence.symbol(), recurrence.launch(), [
      Counts,
      RowOffsets,
      RequestSlots,
      Reset,
      ConvolvedQuery,
      ConvolvedKey,
      ConvolvedValue,
      LogDecay,
      BetaControl,
      RecurrentState,
      DeltaOutput,
    ]),
  )
  let normalization = @norm.rms_norm_source(
    operators.normalization(RecurrentOutput),
    symbol="lunaflux_minicpm_v47_recurrent_norm",
  )
  source += normalization.source_bytes() + b"\n"
  stages.push(
    make(normalization.symbol(), normalization.launch(), [
      Counts,
      DeltaOutput,
      NormWeight,
      Gate,
      GatedOutput,
    ]),
  )
  let projection = operators.linear_projection(Output)
  let symbol = "lunaflux_minicpm_v47_linear_output"
  source += @matrix.dense_source(projection, symbol~)
  stages.push(
    make(symbol, (projection.tile_count(), 1, 128), [
      Counts,
      GatedOutput,
      OutputWeight,
      Output,
    ]),
  )
  let (regions, workspace_bytes) = regions(operators)
  {
    source,
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
  abort("internal recurrent slot")
}
