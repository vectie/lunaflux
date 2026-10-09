///|
/// Emits exact V1 head-reduction source. Layouts are row-major
/// `streams[rows,width,hidden]`, `function_weights[width,width*hidden]`,
/// `base[width]`, scalar `scale[1]`, and `output[rows,hidden]`. Thread zero
/// derives normalized sigmoid controls; all threads then reduce streams in
/// ascending order and round the output once to BF16-RNE.
pub fn head_reduce_v1_bf16_f32_source(
  stream_width~ : Int,
  hidden_width~ : Int,
  maximum_rows~ : Int,
  control_epsilon~ : Double,
  normalization_epsilon~ : Double,
  function_symbol~ : String,
) -> DecoderHyperConnectionHeadReduceCudaSource raise DecoderHyperConnectionCudaSourceError {
  if stream_width <= 1 || stream_width > 16 {
    raise InvalidStreamWidth
  }
  if hidden_width <= 0 || hidden_width > 65536 {
    raise InvalidHiddenWidth
  }
  if maximum_rows <= 0 || maximum_rows > 1048576 {
    raise InvalidMaximumRows
  }
  if control_epsilon.is_nan() ||
    control_epsilon.is_inf() ||
    control_epsilon <= 0.0 ||
    normalization_epsilon.is_nan() ||
    normalization_epsilon.is_inf() ||
    normalization_epsilon <= 0.0 {
    raise InvalidEpsilon
  }
  if !valid_hyper_connection_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let control_literal = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(
    control_epsilon,
  )
  let normalization_literal = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(
    normalization_epsilon,
  )
  let width = stream_width.to_string()
  let hidden = hidden_width.to_string()
  let flattened = (stream_width * hidden_width).to_string()
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n" +
    "__device__ __forceinline__ float lf_hc_head_sigmoid(float value) {\n" +
    "  if (value >= 0.0f) return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-value)));\n" +
    "  const float exponential = expf(value); return __fdiv_rn(exponential, __fadd_rn(1.0f, exponential));\n}\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const __nv_bfloat16 *streams, const float *function_weights, const float *base, const float *scale, __nv_bfloat16 *output) {\n" +
    "  const int32_t live_rows = counts[3];\n" +
    "  if (live_rows < 0 || live_rows > " +
    maximum_rows.to_string() +
    ") return;\n" +
    "  const int32_t row = (int32_t)blockIdx.x;\n" +
    "  if (row >= live_rows) return;\n" +
    "  __shared__ float controls[" +
    width +
    "]; __shared__ int32_t valid;\n" +
    "  const unsigned long long input_base = (unsigned long long)row * " +
    flattened +
    "ULL;\n" +
    "  if (threadIdx.x == 0) { valid = 1; float square_sum = 0.0f; for (int32_t column = 0; column < " +
    flattened +
    "; ++column) { const float value = __bfloat162float(streams[input_base + (unsigned long long)column]); if (!isfinite(value)) { valid = 0; break; } square_sum = __fadd_rn(square_sum, __fmul_rn(value, value)); } if (valid) { const float mean = __fdiv_rn(square_sum, " +
    flattened +
    ".0f); const float inverse = __fdiv_rn(1.0f, sqrtf(__fadd_rn(mean, " +
    normalization_literal +
    "))); const float scalar = scale[0]; if (!isfinite(inverse) || !isfinite(scalar)) valid = 0; for (int32_t stream = 0; stream < " +
    width +
    " && valid; ++stream) { float dot = 0.0f; const unsigned long long weight_base = (unsigned long long)stream * " +
    flattened +
    "ULL; for (int32_t column = 0; column < " +
    flattened +
    "; ++column) { const float weight = function_weights[weight_base + (unsigned long long)column]; if (!isfinite(weight)) { valid = 0; break; } dot = __fadd_rn(dot, __fmul_rn(__bfloat162float(streams[input_base + (unsigned long long)column]), weight)); } const float bias = base[stream]; if (!isfinite(bias)) { valid = 0; break; } const float mix = __fmul_rn(dot, inverse); const float logit = __fadd_rn(__fmul_rn(mix, scalar), bias); controls[stream] = __fadd_rn(lf_hc_head_sigmoid(logit), " +
    control_literal +
    "); if (!isfinite(controls[stream])) valid = 0; } } }\n" +
    "  __syncthreads(); if (!valid) return;\n" +
    "  const unsigned long long output_base = (unsigned long long)row * " +
    hidden +
    "ULL;\n" +
    "  for (int32_t column = (int32_t)threadIdx.x; column < " +
    hidden +
    "; column += (int32_t)blockDim.x) { float sum = 0.0f; for (int32_t stream = 0; stream < " +
    width +
    "; ++stream) { const float value = __bfloat162float(streams[input_base + (unsigned long long)stream * " +
    hidden +
    "ULL + (unsigned long long)column]); sum = __fadd_rn(sum, __fmul_rn(controls[stream], value)); } if (isfinite(sum)) output[output_base + (unsigned long long)column] = __float2bfloat16_rn(sum); }\n" +
    "}\n",
  )
  {
    operation: HeadReduceV1,
    numerical_contract: OrderedF32HeadControlReductionBf16RoundV1,
    stream_width,
    hidden_width,
    maximum_rows,
    control_epsilon,
    normalization_epsilon,
    function_symbol,
    source,
  }
}
