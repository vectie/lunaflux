///|
pub(all) enum Phase {
  Prefill
  Decode
} derive(Eq, Debug)

///|
pub extend Phase with Eq::{equal, not_equal}

///|
pub extend Phase with @debug.Debug::{to_repr}

///|
pub struct Stage {
  priv symbol : String
  priv launch : (Int32, Int32, Int32, Int64)
  priv arguments : ReadOnlyArray[Slot]
  priv attention : Bool
}

///|
pub fn Stage::symbol(self : Self) -> String {
  self.symbol
}

///|
/// grid X/Y, block X, dynamic shared bytes; grid Z and other block axes are one.
pub fn Stage::launch(self : Self) -> (Int32, Int32, Int32, Int64) {
  self.launch
}

///|
pub fn Stage::arguments(self : Self) -> ReadOnlyArray[Slot] {
  self.arguments
}

///|
pub fn Stage::is_attention(self : Self) -> Bool {
  self.attention
}

///|
/// Immutable full-attention startup program. Common sources form one module;
/// prefill/decode remain separate modules to preserve helper namespaces.
/// The input is already decoder-normalized. Residual addition belongs outside.
pub struct Program {
  priv compilation : Compilation
  priv common_source : Bytes
  priv prefill : ReadOnlyArray[Stage]
  priv decode : ReadOnlyArray[Stage]
  priv regions : ReadOnlyArray[Region]
  priv workspace_bytes : Int64
}

///|
pub fn Program::new(
  operators : @operators.DecoderOperators,
  compilation : Compilation,
) -> Self raise {
  let (heads, width, _, rows, _) = operators.query_rotary().geometry()
  let (kv_heads, _, _, _, _) = operators.key_rotary().geometry()
  let (_, _, _, tokens_per_page) = compilation.abi().shape()
  let (row_width, page_stride, pages) = compilation.abi().layout()
  guard compilation.abi().shape() == (heads, kv_heads, width, tokens_per_page) &&
    compilation.abi().limits().1 == rows &&
    row_width == (heads + kv_heads * 2) * width else {
    fail("attention plan/ABI mismatch")
  }
  let make = fn(symbol, launch, arguments) {
    {
      symbol,
      launch,
      arguments: ReadOnlyArray::from_array(arguments),
      attention: false,
    }
  }
  let stages : Array[Stage] = []
  let mut common = b""
  let projections : Array[(String, @operators.FullProjection, Slot, Slot)] = [
    ("q_proj", QueryAndGate, QueryWeight, RawQuery),
    ("k_proj", Key, KeyWeight, RawKey),
    ("v_proj", Value, ValueWeight, RawValue),
  ]
  for (name, site, weight, output) in projections {
    let plan = operators.full_projection(site)
    let symbol = "lunaflux_minicpm_v47_attention_" + name
    common += @matrix.dense_source(plan, symbol~) + b"\n"
    stages.push(
      make(symbol, (plan.tile_count(), 1, 128, 0L), [
        Counts,
        Input,
        weight,
        output,
      ]),
    )
  }
  let norms : Array[(String, @operators.NormSite, Slot, Slot, Slot)] = [
    ("query", FullQuery, RawQuery, QueryNormWeight, Query),
    ("key", FullKey, RawKey, KeyNormWeight, Key),
  ]
  for (name, site, input, weight, output) in norms {
    let source = @norm.rms_norm_source(
      operators.normalization(site),
      symbol="lunaflux_minicpm_v47_attention_" + name + "_norm",
    )
    common += source.source_bytes() + b"\n"
    let (gx, gy, bx) = source.launch()
    stages.push(
      make(source.symbol(), (gx, gy, bx, 0L), [Counts, input, weight, output]),
    )
  }
  let coefficients = @rotary.coefficients_source(
    operators.query_rotary(),
    symbol="lunaflux_minicpm_v47_rotary_coefficients",
  )
  common += coefficients.source_bytes() + b"\n"
  let (gx, gy, bx) = coefficients.launch()
  stages.push(
    make(coefficients.symbol(), (gx, gy, bx, 0L), [
      Counts,
      CanvasPositions,
      Cosine,
      Sine,
    ]),
  )
  for
    (name, plan, tensor) in [
      ("query", operators.query_rotary(), Query),
      ("key", operators.key_rotary(), Key),
    ] {
    let source = @rotary.rotary_source(
      plan,
      symbol="lunaflux_minicpm_v47_" + name + "_rotary",
    )
    common += source.source_bytes() + b"\n"
    let (gx, gy, bx) = source.launch()
    stages.push(
      make(source.symbol(), (gx, gy, bx, 0L), [
        Counts,
        tensor,
        Cosine,
        Sine,
        tensor,
      ]),
    )
  }
  let packed = @precision.PackedQkvPrecision::new(
    query_width=heads * width,
    kv_width=kv_heads * width,
    maximum_rows=rows,
  )
  common += @pack.packed_qkv_source(
      packed,
      symbol="lunaflux_minicpm_v47_pack_qkv",
    ) +
    b"\n"
  stages.push(
    make("lunaflux_minicpm_v47_pack_qkv", (rows, 1, 128, 0L), [
      Counts,
      Query,
      Key,
      RawValue,
      PackedQkv,
    ]),
  )
  guard @effects.KvWriteProgram::new(
      query_heads=heads,
      kv_heads~,
      head_dimension=width,
      page_stride_bytes=page_stride * 2L,
      element_bytes=2,
      owners=128,
    )
    is Some(writer) else {
    fail("unsupported KV writer geometry")
  }
  common += @pointwise.paged_kv_write_source(
      writer,
      symbol="lunaflux_minicpm_v47_write_kv",
      tokens_per_page~,
      physical_page_capacity=pages.to_int(),
    ) +
    b"\n"
  stages.push(
    make(
      "lunaflux_minicpm_v47_write_kv",
      (
        ((rows.to_int64() * kv_heads.to_int64() * width.to_int64() + 127L) /
        128L).to_int(),
        1,
        128,
        0L,
      ),
      [
        Counts,
        CausalPositions,
        RowOffsets,
        SequenceLengths,
        PageOffsets,
        PhysicalPages,
        PackedQkv,
        KeyCache,
        ValueCache,
      ],
    ),
  )
  let attention_arguments = [
    Counts,
    CausalPositions,
    RowOffsets,
    SequenceLengths,
    PageOffsets,
    PhysicalPages,
    PackedQkv,
    Context,
    KeyCache,
    ValueCache,
  ]
  let prefill_stage = {
    ..make(
      compilation.prefill().source().symbol().to_owned(),
      compilation.prefill().lowering().launch(),
      attention_arguments,
    ),
    attention: true,
  }
  let decode_stage = {
    ..make(
      compilation.decode().source().symbol().to_owned(),
      compilation.decode().lowering().launch(),
      attention_arguments,
    ),
    attention: true,
  }
  let gate = @gate.pointwise_gate_source(
    operators.full_output_gate(),
    symbol="lunaflux_minicpm_v47_attention_gate",
  )
  common += gate.source_bytes() + b"\n"
  let (gx, gy, bx) = gate.launch()
  let gate_stage = make(gate.symbol(), (gx, gy, bx, 0L), [
    Counts,
    Context,
    RawQuery,
    GatedContext,
  ])
  let output = operators.full_projection(Output)
  common += @matrix.dense_source(
    output,
    symbol="lunaflux_minicpm_v47_attention_o_proj",
  )
  let output_stage = make(
    "lunaflux_minicpm_v47_attention_o_proj",
    (output.tile_count(), 1, 128, 0L),
    [Counts, GatedContext, OutputWeight, Output],
  )
  let prefill = stages.copy()
  prefill.push(prefill_stage)
  prefill.push(gate_stage)
  prefill.push(output_stage)
  let decode = stages.copy()
  decode.push(decode_stage)
  decode.push(gate_stage)
  decode.push(output_stage)
  let (regions, workspace_bytes) = regions(operators, compilation)
  {
    compilation,
    common_source: common,
    prefill: ReadOnlyArray::from_array(prefill),
    decode: ReadOnlyArray::from_array(decode),
    regions,
    workspace_bytes,
  }
}

///|
pub fn Program::common_source_bytes(self : Self) -> Bytes {
  self.common_source
}

///|
pub fn Program::attention_source_bytes(self : Self, phase : Phase) -> Bytes {
  match phase {
    Prefill => self.compilation.prefill().source().bytes()
    Decode => self.compilation.decode().source().bytes()
  }
}

///|
pub fn Program::stages(self : Self, phase : Phase) -> ReadOnlyArray[Stage] {
  match phase {
    Prefill => self.prefill
    Decode => self.decode
  }
}

///|
pub fn Program::workspace_bytes(self : Self) -> Int64 {
  self.workspace_bytes
}

///|
pub fn Program::abi(self : Self) -> @source.CudaPagedAttentionTileAbi {
  self.compilation.abi()
}

///|
pub fn Program::context_tokens(self : Self) -> Int32 {
  self.compilation.context_tokens()
}

///|
pub fn Program::regions(self : Self) -> ReadOnlyArray[Region] {
  self.regions
}

///|
pub fn Program::region(self : Self, slot : Slot) -> Region {
  for region in self.regions {
    if region.slot == slot {
      return region
    }
  }
  abort("internal attention slot")
}
