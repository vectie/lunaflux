///|
pub(all) suberror HyperConnectionMixCudaSourceError {
  InvalidHiddenWidth
  InvalidStreamCount
  InvalidMaximumRows
  InvalidFunctionSymbol
  DimensionOverflow
} derive(Debug, Eq)

///|
pub(all) enum HyperConnectionMixNumericalContract {
  F32ControlsRoundedBf16OrderedMixV1
} derive(Debug, Eq)

///|
pub struct HyperConnectionMixCudaSource {
  priv hidden_width : Int
  priv stream_count : Int
  priv maximum_rows : Int
  priv function_symbol : String
  priv source : Bytes
}

///|
pub fn hyper_connection_mix_bf16_source(
  hidden_width~ : Int,
  stream_count~ : Int,
  maximum_rows~ : Int,
  function_symbol~ : String,
) -> HyperConnectionMixCudaSource raise HyperConnectionMixCudaSourceError {
  if hidden_width <= 0 {
    raise InvalidHiddenWidth
  }
  if stream_count <= 0 || stream_count > 64 {
    raise InvalidStreamCount
  }
  if maximum_rows <= 0 || maximum_rows > 1048576 {
    raise InvalidMaximumRows
  }
  let cells = maximum_rows.to_int64() * stream_count.to_int64()
  if cells <= 0L || hidden_width.to_int64() > 9223372036854775552L / cells {
    raise DimensionOverflow
  }
  if function_symbol.is_empty() ||
    function_symbol.length() > 128 ||
    !(function_symbol[0] is ('a'..='z' | 'A'..='Z' | '_')) ||
    !function_symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_')) {
    raise InvalidFunctionSymbol
  }
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const float *post, const float *combine, const __nv_bfloat16 *residual, const __nv_bfloat16 *branch, __nv_bfloat16 *output) {\n" +
    "  if (output == residual || output == branch) return; const int32_t live_rows = counts[3]; if (live_rows <= 0 || live_rows > " +
    maximum_rows.to_string() +
    ") return;\n" +
    "  const unsigned long long index = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x; const unsigned long long live = (unsigned long long)live_rows * " +
    stream_count.to_string() +
    "ULL * " +
    hidden_width.to_string() +
    "ULL; if (index >= live) return;\n" +
    "  const unsigned long long component = index % " +
    hidden_width.to_string() +
    "ULL; const unsigned long long stream_cell = index / " +
    hidden_width.to_string() +
    "ULL; const unsigned long long destination = stream_cell % " +
    stream_count.to_string() +
    "ULL; const unsigned long long row = stream_cell / " +
    stream_count.to_string() +
    "ULL;\n" +
    "  const __nv_bfloat16 post_bf16 = __float2bfloat16_rn(post[row * " +
    stream_count.to_string() +
    "ULL + destination]); const __nv_bfloat16 branch_product = __float2bfloat16_rn(__fmul_rn(__bfloat162float(post_bf16), __bfloat162float(branch[row * " +
    hidden_width.to_string() +
    "ULL + component])));\n" +
    "  float residual_sum = 0.0f; for (int source = 0; source < " +
    stream_count.to_string() +
    "; ++source) { const __nv_bfloat16 coefficient = __float2bfloat16_rn(combine[(row * " +
    stream_count.to_string() +
    "ULL + (unsigned long long)source) * " +
    stream_count.to_string() +
    "ULL + destination]); const float product = __fmul_rn(__bfloat162float(coefficient), __bfloat162float(residual[(row * " +
    stream_count.to_string() +
    "ULL + (unsigned long long)source) * " +
    hidden_width.to_string() +
    "ULL + component])); residual_sum = __fadd_rn(residual_sum, product); }\n" +
    "  const __nv_bfloat16 mixed_residual = __float2bfloat16_rn(residual_sum); output[index] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(branch_product), __bfloat162float(mixed_residual)));\n}\n",
  )
  { hidden_width, stream_count, maximum_rows, function_symbol, source, }
}

///|
pub fn HyperConnectionMixCudaSource::hidden_width(self : Self) -> Int {
  self.hidden_width
}

///|
pub fn HyperConnectionMixCudaSource::stream_count(self : Self) -> Int {
  self.stream_count
}

///|
pub fn HyperConnectionMixCudaSource::maximum_rows(self : Self) -> Int {
  self.maximum_rows
}

///|
pub fn HyperConnectionMixCudaSource::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn HyperConnectionMixCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn HyperConnectionMixCudaSource::numerical_contract(
  _self : Self,
) -> HyperConnectionMixNumericalContract {
  F32ControlsRoundedBf16OrderedMixV1
}
