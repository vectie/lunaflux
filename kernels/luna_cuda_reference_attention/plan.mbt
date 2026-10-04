///|
/// CUDA-only terminal choice. Model and scheduling plans do not name vendors.
pub(all) enum ReferenceAttentionBackend {
  FlashAttentionPrefill
  FlashAttentionDecode
  FlashInferDecode
} derive(Debug, Eq)

///|
/// Immutable AOT adapter; no library discovery or planning occurs in a step.
pub struct ReferenceAttentionPlan {
  priv backend : ReferenceAttentionBackend
  priv abi : @abi.CudaPagedAttentionTileAbi
  priv symbol : String
  priv query_dtype : @numeric.ComputeDType
  priv kv_dtype : @numeric.ComputeDType
  priv output_dtype : @numeric.ComputeDType
  priv accumulator : @numeric.AccumulatorDType
} derive(Debug, Eq)

///|
pub(all) suberror ReferenceAttentionError {
  UnsupportedReferenceAttention
  UnsupportedReferenceAttentionFormat
} derive(Debug, Eq)

///|
/// Read-only attention must follow a separately committed KV-write effect.
/// This is not an alternative for the legacy write-and-attend ABI.
pub fn ReferenceAttentionPlan::new(
  backend~ : ReferenceAttentionBackend,
  abi~ : @abi.CudaPagedAttentionTileAbi,
  query_dtype? : @numeric.ComputeDType = @numeric.ComputeDType::bfloat16(),
  kv_dtype? : @numeric.ComputeDType = @numeric.ComputeDType::bfloat16(),
  output_dtype? : @numeric.ComputeDType = @numeric.ComputeDType::bfloat16(),
  accumulator? : @numeric.AccumulatorDType = @numeric.AccumulatorDType::f32(),
) -> ReferenceAttentionPlan raise ReferenceAttentionError {
  // Resolve the terminal specialization at startup, never by casting a
  // quantized cache to BF16. Weight dtype is deliberately not an attention
  // input: upstream projection owns its storage, scales and conversion.
  if query_dtype != @numeric.ComputeDType::bfloat16() ||
    kv_dtype != @numeric.ComputeDType::bfloat16() ||
    output_dtype != @numeric.ComputeDType::bfloat16() ||
    accumulator != @numeric.AccumulatorDType::f32() {
    raise UnsupportedReferenceAttentionFormat
  }
  let (heads, kv_heads, dimension, page) = abi.shape()
  let (width, stride, _) = abi.layout()
  if dimension != 128 ||
    !((heads / kv_heads) is (1 | 2 | 4 | 8)) ||
    kv_heads.to_int64() >
    9223372036854775807L / dimension.to_int64() / page.to_int64() ||
    stride > 2147483647L ||
    width.to_int64() <
    (heads.to_int64() + 2L * kv_heads.to_int64()) * dimension.to_int64() ||
    stride != page.to_int64() * kv_heads.to_int64() * dimension.to_int64() ||
    abi.query_metadata_version() != 0 {
    raise UnsupportedReferenceAttention
  }
  let symbol = match backend {
    FlashAttentionPrefill =>
      "lunaflux_attention_prefill_flashattention_bf16_exp2_v1"
    FlashAttentionDecode =>
      "lunaflux_attention_decode_flashattention_bf16_exp2_v1"
    FlashInferDecode => "lunaflux_attention_decode_flashinfer_f32_exp2_v1"
  }
  { backend, abi, symbol, query_dtype, kv_dtype, output_dtype, accumulator, }
}

///|
pub fn ReferenceAttentionPlan::operand_formats(
  self : ReferenceAttentionPlan,
) -> (
  @numeric.ComputeDType,
  @numeric.ComputeDType,
  @numeric.ComputeDType,
  @numeric.AccumulatorDType,
) {
  (self.query_dtype, self.kv_dtype, self.output_dtype, self.accumulator)
}

///|
pub fn ReferenceAttentionPlan::symbol(self : ReferenceAttentionPlan) -> String {
  self.symbol
}

///|
pub fn ReferenceAttentionPlan::numeric_law(
  self : ReferenceAttentionPlan,
) -> String {
  match self.backend {
    FlashAttentionPrefill | FlashAttentionDecode =>
      "reference-flashattention-bf16-probability-exp2-v1"
    FlashInferDecode => "reference-flashinfer-f32-probability-exp2-v1"
  }
}

///|
/// Block geometry is the reference's explicit specialization, not a generic
/// compiler heuristic. Grid X covers the same row-major tile domain.
pub fn ReferenceAttentionPlan::launch(
  self : ReferenceAttentionPlan,
) -> (Int, Int, Int, Int, Int, Int, Int) {
  let (heads, kv_heads, _, _) = self.abi.shape()
  let (rows, tokens, _) = self.abi.limits()
  match self.backend {
    FlashAttentionPrefill =>
      (
        rows +
        (tokens - rows) / 64 +
        (if (tokens - rows) % 64 == 0 { 0 } else { 1 }),
        heads,
        1,
        128,
        1,
        1,
        81920,
      )
    FlashAttentionDecode => (rows, kv_heads, 1, 128, 1, 1, 81920)
    FlashInferDecode => {
      let group = heads / kv_heads
      let z = 128 / (16 * group)
      let shared = 2 * 2 * group * z * 128 * 2 + 128 * 8
      (rows, kv_heads, 1, 16, group, z, shared)
    }
  }
}

///|
pub fn ReferenceAttentionPlan::source(self : ReferenceAttentionPlan) -> Bytes {
  @utf8.encode(
    match self.backend {
      FlashAttentionPrefill | FlashAttentionDecode =>
        flashattention_source(self)
      FlashInferDecode => flashinfer_source(self)
    },
  )
}

///|
pub fn ReferenceAttentionPlan::source_sha256(
  self : ReferenceAttentionPlan,
) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(self.source()))
}
