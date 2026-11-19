///|
pub(all) enum Slot {
  Counts
  RowOffsets
  RequestSlots
  Reset
  Input
  Output
  QkvWeight
  GateWeight
  DecayWeight
  BetaWeight
  OutputWeight
  ConvolutionWeight
  NormWeight
  ALog
  DtBias
  ConvolutionState
  RecurrentState
  RawQkv
  Gate
  Decay
  Beta
  ConvolvedQuery
  ConvolvedKey
  ConvolvedValue
  LogDecay
  BetaControl
  DeltaOutput
  GatedOutput
} derive(Eq, Debug)

///|
pub extend Slot with Eq::{equal, not_equal}

///|
pub extend Slot with @debug.Debug::{to_repr}

///|
pub fn Slot::name(self : Self) -> String {
  match self {
    Counts => "counts"
    RowOffsets => "row_offsets"
    RequestSlots => "request_slots"
    Reset => "reset"
    Input => "input"
    Output => "output"
    QkvWeight => "qkv_weight"
    GateWeight => "gate_weight"
    DecayWeight => "decay_weight"
    BetaWeight => "beta_weight"
    OutputWeight => "output_weight"
    ConvolutionWeight => "convolution_weight"
    NormWeight => "norm_weight"
    ALog => "a_log"
    DtBias => "dt_bias"
    ConvolutionState => "convolution_state"
    RecurrentState => "recurrent_state"
    RawQkv => "raw_qkv"
    Gate => "gate"
    Decay => "decay"
    Beta => "beta"
    ConvolvedQuery => "convolved_query"
    ConvolvedKey => "convolved_key"
    ConvolvedValue => "convolved_value"
    LogDecay => "log_decay"
    BetaControl => "beta_control"
    DeltaOutput => "delta_output"
    GatedOutput => "gated_output"
  }
}

///|
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
  let delta = operators.recurrent()
  let conv = operators.convolution()
  let (heads, keys, values) = delta.geometry()
  let (rows, sequences, _) = delta.capacity()
  let (channels, kernel) = conv.geometry()
  let (_, hidden, _) = operators.linear_projection(Output).geometry()
  let row = rows.to_int64()
  let qk = delta.query_key_head_count().to_int64() * keys.to_int64()
  let vw = heads.to_int64() * values.to_int64()
  let weight_bytes = fn(site) raise {
    let (_, columns, inner) = operators.linear_projection(site).geometry()
    columns.to_int64() * inner.to_int64() * 2L
  }
  let borrowed : Array[(Slot, Int64, Int64)] = [
    (Counts, 20L, 4L),
    (RowOffsets, (sequences + 1).to_int64() * 4L, 4L),
    (RequestSlots, sequences.to_int64() * 4L, 4L),
    (Reset, sequences.to_int64() * 4L, 4L),
    (Input, row * hidden.to_int64() * 2L, 2L),
    (Output, row * hidden.to_int64() * 2L, 2L),
    (QkvWeight, weight_bytes(QueryKeyValue), 2L),
    (GateWeight, weight_bytes(Gate), 2L),
    (DecayWeight, weight_bytes(Decay), 2L),
    (BetaWeight, weight_bytes(Beta), 2L),
    (OutputWeight, weight_bytes(Output), 2L),
    (ConvolutionWeight, channels.to_int64() * kernel.to_int64() * 2L, 2L),
    (NormWeight, values.to_int64() * 2L, 2L),
    (ALog, heads.to_int64() * 2L, 2L),
    (DtBias, heads.to_int64() * 2L, 2L),
    (ConvolutionState, conv.state_bytes(), 2L),
    (RecurrentState, delta.state_bytes(), 4L),
  ]
  let scratch : Array[(Slot, Int64, Int64)] = [
    (RawQkv, row * channels.to_int64() * 2L, 2L),
    (Gate, row * vw * 2L, 2L),
    (Decay, row * heads.to_int64() * 2L, 2L),
    (Beta, row * heads.to_int64() * 2L, 2L),
    (ConvolvedQuery, row * channels.to_int64() * 2L, 2L),
    (LogDecay, row * heads.to_int64() * 4L, 4L),
    (BetaControl, row * heads.to_int64() * 2L, 2L),
    (DeltaOutput, row * vw * 2L, 2L),
    (GatedOutput, row * vw * 2L, 2L),
  ]
  let out : Array[Region] = borrowed.map(entry => {
    let (slot, byte_count, alignment) = entry
    { slot, workspace_offset: None, byte_count, alignment, }
  })
  let mut offset = 0L
  let mut convolved = 0L
  for (slot, bytes, alignment) in scratch {
    guard offset <= 9223372036854775792L &&
      bytes <= 9223372036854775807L - (offset + 15L) / 16L * 16L else {
      fail("recurrent workspace overflow")
    }
    offset = (offset + 15L) / 16L * 16L
    out.push({
      slot,
      workspace_offset: Some(offset),
      byte_count: bytes,
      alignment,
    })
    if slot is ConvolvedQuery {
      convolved = offset
    }
    offset += bytes
  }
  // Pointer views into one convolved packed row bank, not extra allocations.
  let stride = channels.to_int64()
  out.push({
    slot: ConvolvedKey,
    workspace_offset: Some(convolved + qk * 2L),
    byte_count: ((row - 1L) * stride + qk) * 2L,
    alignment: 2L,
  })
  out.push({
    slot: ConvolvedValue,
    workspace_offset: Some(convolved + qk * 4L),
    byte_count: ((row - 1L) * stride + vw) * 2L,
    alignment: 2L,
  })
  (ReadOnlyArray::from_array(out), offset)
}
