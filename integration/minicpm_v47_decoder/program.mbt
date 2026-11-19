///|
pub using @attention {type Phase}

///|
pub(all) enum Operation {
  NormalizeInput
  Branch
  AddBranchResidual
  NormalizeMoe
  Moe
  AddMoeResidual
} derive(Eq, Debug)

///|
pub extend Operation with Eq::{equal, not_equal}

///|
pub extend Operation with @debug.Debug::{to_repr}

///|
pub fn Operation::name(self : Self) -> String {
  match self {
    NormalizeInput => "normalize_input"
    Branch => "branch"
    AddBranchResidual => "add_branch_residual"
    NormalizeMoe => "normalize_moe"
    Moe => "moe"
    AddMoeResidual => "add_moe_residual"
  }
}

///|
/// state_offset addresses either the FP32 recurrent arena or each BF16 KV plane.
pub(all) struct LayerPlan {
  index : Int32
  kind : @hybrid.LayerKind
  state_offset : Int64
  convolution_offset : Int64?
}

///|
pub(all) struct BufferBudget {
  activation_bytes : Int64
  branch_workspace_bytes : Int64
  moe_workspace_bytes : Int64
  convolution_state_bytes : Int64
  recurrent_state_bytes : Int64
  key_cache_bytes : Int64
  value_cache_bytes : Int64
}

///|
pub struct Program {
  priv attention : @attention.Program
  priv recurrent : @recurrent.Program
  priv moe : @moe.Program
  priv source : Bytes
  priv norm_symbol : String
  priv residual_symbol : String
  priv norm_launch : (Int32, Int32, Int32)
  priv residual_launch : (Int32, Int32, Int32)
  priv layers : ReadOnlyArray[LayerPlan]
  priv operations : ReadOnlyArray[Operation]
  priv budget : BufferBudget
}

///|
pub fn Program::new(
  operators : @operators.DecoderOperators,
  compilation : @attention.Compilation,
) -> Self raise {
  let attention = @attention.Program::new(operators, compilation)
  let recurrent = @recurrent.Program::new(operators)
  let moe = @moe.Program::new(operators)
  guard attention.region(RowOffsets).byte_count ==
    recurrent.region(RowOffsets).byte_count &&
    attention.region(Input).byte_count == recurrent.region(Input).byte_count &&
    moe.region(Input).byte_count == recurrent.region(Input).byte_count else {
    fail("decoder branch capacity mismatch")
  }
  let normalization = @norm.rms_norm_source(
    operators.normalization(Decoder),
    symbol="lunaflux_minicpm_v47_decoder_norm",
  )
  let (width, _, rows) = operators.normalization(Decoder).geometry()
  guard @physical.VectorOwnership::new(vectors=width, owners=256)
    is Some(ownership) else {
    fail("decoder residual ownership")
  }
  let residual_symbol = "lunaflux_minicpm_v47_decoder_residual"
  let residual = @pointwise.bf16_residual_source(
    ownership,
    symbol=residual_symbol,
  )
  let geometry = operators.geometry()
  let recurrent_bytes = recurrent.region(RecurrentState).byte_count
  let convolution_bytes = recurrent.region(ConvolutionState).byte_count
  let cache_bytes = attention.region(KeyCache).byte_count
  let layers : Array[LayerPlan] = []
  let mut recurrent_count = 0L
  let mut full_count = 0L
  for index in 0..<geometry.layer_count() {
    let kind = geometry.layer_kind(index).unwrap()
    match kind {
      GatedDeltaLinearAttention => {
        layers.push({
          index,
          kind,
          state_offset: recurrent_count * recurrent_bytes,
          convolution_offset: Some(recurrent_count * convolution_bytes),
        })
        recurrent_count += 1L
      }
      FullAttention => {
        layers.push({
          index,
          kind,
          state_offset: full_count * cache_bytes,
          convolution_offset: None,
        })
        full_count += 1L
      }
    }
  }
  {
    attention,
    recurrent,
    moe,
    source: normalization.source_bytes() + b"\n" + residual,
    norm_symbol: normalization.symbol(),
    residual_symbol,
    norm_launch: normalization.launch(),
    residual_launch: (
      ((rows.to_int64() * width.to_int64() + 255L) / 256L).to_int(),
      1,
      256,
    ),
    layers: ReadOnlyArray::from_array(layers),
    operations: [
      NormalizeInput,
      Branch,
      AddBranchResidual,
      NormalizeMoe,
      Moe,
      AddMoeResidual,
    ],
    budget: {
      activation_bytes: recurrent.region(Input).byte_count,
      branch_workspace_bytes: recurrent
      .workspace_bytes()
      .max(attention.workspace_bytes()),
      moe_workspace_bytes: moe.workspace_bytes(),
      convolution_state_bytes: recurrent_count * convolution_bytes,
      recurrent_state_bytes: recurrent_count * recurrent_bytes,
      key_cache_bytes: full_count * cache_bytes,
      value_cache_bytes: full_count * cache_bytes,
    },
  }
}

///|
pub fn Program::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn Program::layers(self : Self) -> ReadOnlyArray[LayerPlan] {
  self.layers
}

///|
pub fn Program::operations(self : Self) -> ReadOnlyArray[Operation] {
  self.operations
}

///|
pub fn Program::budget(self : Self) -> BufferBudget {
  self.budget
}

///|
pub fn Program::attention_program(self : Self) -> @attention.Program {
  self.attention
}

///|
pub fn Program::recurrent_program(self : Self) -> @recurrent.Program {
  self.recurrent
}

///|
pub fn Program::moe_program(self : Self) -> @moe.Program {
  self.moe
}

///|
pub fn Program::launch_count(self : Self, phase : Phase) -> Int32 {
  let mut count = 0
  for layer in self.layers {
    let branch_count = match layer.kind {
      GatedDeltaLinearAttention => self.recurrent.stages().length()
      FullAttention => self.attention.stages(phase).length()
    }
    count += 4 + self.moe.stages().length() + branch_count
  }
  count
}
