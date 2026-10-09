///|
fn valid_hyper_connection_symbol(symbol : StringView) -> Bool {
  if symbol.is_empty() || symbol.length() > 128 {
    return false
  }
  symbol[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
/// Emits exact V1 block-control source. Output layouts are row-major
/// `pre[rows,width]`, `post[rows,width]`, and
/// `combination[rows,width,width]`, with destination before source.
pub fn block_control_v1_bf16_f32_source(
  stream_width~ : Int,
  hidden_width~ : Int,
  maximum_rows~ : Int,
  sinkhorn_iterations~ : Int,
  control_epsilon~ : Double,
  normalization_epsilon~ : Double,
  function_symbol~ : String,
) -> DecoderHyperConnectionCudaSource raise DecoderHyperConnectionCudaSourceError {
  if stream_width <= 1 || stream_width > 16 {
    raise InvalidStreamWidth
  }
  if hidden_width <= 0 || hidden_width > 65536 {
    raise InvalidHiddenWidth
  }
  if maximum_rows <= 0 || maximum_rows > 1048576 {
    raise InvalidMaximumRows
  }
  if sinkhorn_iterations <= 0 || sinkhorn_iterations > 64 {
    raise InvalidIterationCount
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
  let flattened = (stream_width * hidden_width).to_string()
  let controls = (6 * stream_width).to_string()
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n" +
    "__device__ __forceinline__ float lf_hc_sigmoid(float value) {\n" +
    "  if (value >= 0.0f) return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-value)));\n" +
    "  const float exponential = expf(value); return __fdiv_rn(exponential, __fadd_rn(1.0f, exponential));\n}\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const __nv_bfloat16 *streams, const float *function_weights, const float *base, const float *scale, float *pre, float *post, float *combination) {\n" +
    "  const int32_t live_rows = counts[3];\n" +
    "  if (live_rows < 0 || live_rows > " +
    maximum_rows.to_string() +
    ") return;\n" +
    "  const int32_t row = (int32_t)blockIdx.x;\n" +
    "  if (row >= live_rows || threadIdx.x != 0) return;\n" +
    "  const unsigned long long input_base = (unsigned long long)row * " +
    flattened +
    "ULL;\n" +
    "  float square_sum = 0.0f;\n" +
    "  for (int32_t column = 0; column < " +
    flattened +
    "; ++column) { const float value = __bfloat162float(streams[input_base + (unsigned long long)column]); if (!isfinite(value)) return; square_sum = __fadd_rn(square_sum, __fmul_rn(value, value)); }\n" +
    "  const float mean = __fdiv_rn(square_sum, " +
    flattened +
    ".0f);\n" +
    "  const float inverse = __fdiv_rn(1.0f, sqrtf(__fadd_rn(mean, " +
    normalization_literal +
    "))); if (!isfinite(inverse)) return;\n" +
    "  float mixes[" +
    controls +
    "];\n" +
    "  for (int32_t control = 0; control < " +
    controls +
    "; ++control) { float dot = 0.0f; const unsigned long long weight_base = (unsigned long long)control * " +
    flattened +
    "ULL; for (int32_t column = 0; column < " +
    flattened +
    "; ++column) { const float weight = function_weights[weight_base + (unsigned long long)column]; if (!isfinite(weight)) return; dot = __fadd_rn(dot, __fmul_rn(__bfloat162float(streams[input_base + (unsigned long long)column]), weight)); } mixes[control] = __fmul_rn(dot, inverse); if (!isfinite(mixes[control])) return; }\n" +
    "  for (int32_t control = 0; control < " +
    controls +
    "; ++control) if (!isfinite(base[control])) return;\n" +
    "  for (int32_t index = 0; index < 3; ++index) if (!isfinite(scale[index])) return;\n" +
    "  for (int32_t control = 0; control < " +
    controls +
    "; ++control) { const int32_t group = control < " +
    width +
    " ? 0 : (control < 2 * " +
    width +
    " ? 1 : 2); if (!isfinite(__fadd_rn(__fmul_rn(mixes[control], scale[group]), base[control]))) return; }\n" +
    "  const unsigned long long vector_base = (unsigned long long)row * " +
    width +
    "ULL;\n" +
    "  for (int32_t stream = 0; stream < " +
    width +
    "; ++stream) { const float pre_logit = __fadd_rn(__fmul_rn(mixes[stream], scale[0]), base[stream]); pre[vector_base + (unsigned long long)stream] = __fadd_rn(lf_hc_sigmoid(pre_logit), " +
    control_literal +
    "); const int32_t post_index = " +
    width +
    " + stream; const float post_logit = __fadd_rn(__fmul_rn(mixes[post_index], scale[1]), base[post_index]); post[vector_base + (unsigned long long)stream] = __fmul_rn(2.0f, lf_hc_sigmoid(post_logit)); }\n" +
    "  const unsigned long long matrix_base = (unsigned long long)row * " +
    width +
    "ULL * " +
    width +
    "ULL;\n" +
    "  for (int32_t destination = 0; destination < " +
    width +
    "; ++destination) { int32_t control = 2 * " +
    width +
    " + destination * " +
    width +
    "; float maximum = __fadd_rn(__fmul_rn(mixes[control], scale[2]), base[control]); for (int32_t source = 1; source < " +
    width +
    "; ++source) { control = 2 * " +
    width +
    " + destination * " +
    width +
    " + source; const float value = __fadd_rn(__fmul_rn(mixes[control], scale[2]), base[control]); if (value > maximum) maximum = value; } float sum = 0.0f; for (int32_t source = 0; source < " +
    width +
    "; ++source) { control = 2 * " +
    width +
    " + destination * " +
    width +
    " + source; const float logit = __fadd_rn(__fmul_rn(mixes[control], scale[2]), base[control]); const float value = expf(__fadd_rn(logit, -maximum)); const unsigned long long index = matrix_base + (unsigned long long)destination * " +
    width +
    "ULL + (unsigned long long)source; combination[index] = value; sum = __fadd_rn(sum, value); } for (int32_t source = 0; source < " +
    width +
    "; ++source) { const unsigned long long index = matrix_base + (unsigned long long)destination * " +
    width +
    "ULL + (unsigned long long)source; combination[index] = __fadd_rn(__fdiv_rn(combination[index], sum), " +
    control_literal +
    "); } }\n" +
    "  for (int32_t iteration = 0; iteration < " +
    sinkhorn_iterations.to_string() +
    "; ++iteration) { if (iteration > 0) { for (int32_t destination = 0; destination < " +
    width +
    "; ++destination) { float sum = 0.0f; for (int32_t source = 0; source < " +
    width +
    "; ++source) sum = __fadd_rn(sum, combination[matrix_base + (unsigned long long)destination * " +
    width +
    "ULL + (unsigned long long)source]); const float denominator = __fadd_rn(sum, " +
    control_literal +
    "); for (int32_t source = 0; source < " +
    width +
    "; ++source) { const unsigned long long index = matrix_base + (unsigned long long)destination * " +
    width +
    "ULL + (unsigned long long)source; combination[index] = __fdiv_rn(combination[index], denominator); } } } for (int32_t source = 0; source < " +
    width +
    "; ++source) { float sum = 0.0f; for (int32_t destination = 0; destination < " +
    width +
    "; ++destination) sum = __fadd_rn(sum, combination[matrix_base + (unsigned long long)destination * " +
    width +
    "ULL + (unsigned long long)source]); const float denominator = __fadd_rn(sum, " +
    control_literal +
    "); for (int32_t destination = 0; destination < " +
    width +
    "; ++destination) { const unsigned long long index = matrix_base + (unsigned long long)destination * " +
    width +
    "ULL + (unsigned long long)source; combination[index] = __fdiv_rn(combination[index], denominator); } } }\n" +
    "}\n",
  )
  {
    operation: BlockControlV1,
    numerical_contract: OrderedF32BlockControlV1,
    stream_width,
    hidden_width,
    maximum_rows,
    sinkhorn_iterations,
    control_epsilon,
    normalization_epsilon,
    function_symbol,
    source,
  }
}

///|
/// Emits exact V1 pre-reduction source. Layouts are row-major
/// `streams[rows,width,hidden]`, `pre[rows,width]`, and
/// `output[rows,hidden]`. Each output element rounds once to BF16-RNE.
pub fn pre_reduce_v1_bf16_source(
  stream_width~ : Int,
  hidden_width~ : Int,
  maximum_rows~ : Int,
  function_symbol~ : String,
) -> DecoderHyperConnectionPreReduceCudaSource raise DecoderHyperConnectionCudaSourceError {
  if stream_width <= 1 || stream_width > 16 {
    raise InvalidStreamWidth
  }
  if hidden_width <= 0 || hidden_width > 65536 {
    raise InvalidHiddenWidth
  }
  if maximum_rows <= 0 || maximum_rows > 1048576 {
    raise InvalidMaximumRows
  }
  if !valid_hyper_connection_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let width = stream_width.to_string()
  let hidden = hidden_width.to_string()
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const __nv_bfloat16 *streams, const float *pre, __nv_bfloat16 *output) {\n" +
    "  const int32_t live_rows = counts[3];\n" +
    "  if (live_rows < 0 || live_rows > " +
    maximum_rows.to_string() +
    ") return;\n" +
    "  const int32_t row = (int32_t)blockIdx.x;\n" +
    "  if (row >= live_rows) return;\n" +
    "  const unsigned long long stream_row_base = (unsigned long long)row * " +
    width +
    "ULL * " +
    hidden +
    "ULL;\n" +
    "  const unsigned long long pre_base = (unsigned long long)row * " +
    width +
    "ULL;\n" +
    "  const unsigned long long output_base = (unsigned long long)row * " +
    hidden +
    "ULL;\n" +
    "  for (int32_t column = (int32_t)threadIdx.x; column < " +
    hidden +
    "; column += (int32_t)blockDim.x) { float sum = 0.0f; bool valid = true; for (int32_t stream = 0; stream < " +
    width +
    "; ++stream) { const float control = pre[pre_base + (unsigned long long)stream]; const float value = __bfloat162float(streams[stream_row_base + (unsigned long long)stream * " +
    hidden +
    "ULL + (unsigned long long)column]); if (!isfinite(control) || !isfinite(value)) { valid = false; break; } sum = __fadd_rn(sum, __fmul_rn(control, value)); } if (valid && isfinite(sum)) output[output_base + (unsigned long long)column] = __float2bfloat16_rn(sum); }\n" +
    "}\n",
  )
  {
    operation: PreReduceV1,
    numerical_contract: OrderedF32PreReduceBf16RoundV1,
    stream_width,
    hidden_width,
    maximum_rows,
    function_symbol,
    source,
  }
}

///|
/// Emits exact V1 post-combination source. Layouts are row-major
/// `branch[rows,hidden]`, `residual[rows,width,hidden]`,
/// `post[rows,width]`, `combination[rows,width,width]`, and
/// `output[rows,width,hidden]`, with destination before source.
pub fn post_combine_v1_bf16_source(
  stream_width~ : Int,
  hidden_width~ : Int,
  maximum_rows~ : Int,
  function_symbol~ : String,
) -> DecoderHyperConnectionPostCombineCudaSource raise DecoderHyperConnectionCudaSourceError {
  if stream_width <= 1 || stream_width > 16 {
    raise InvalidStreamWidth
  }
  if hidden_width <= 0 || hidden_width > 65536 {
    raise InvalidHiddenWidth
  }
  if maximum_rows <= 0 || maximum_rows > 1048576 {
    raise InvalidMaximumRows
  }
  if !valid_hyper_connection_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let width = stream_width.to_string()
  let hidden = hidden_width.to_string()
  let flattened = (stream_width * hidden_width).to_string()
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const __nv_bfloat16 *branch, const __nv_bfloat16 *residual, const float *post, const float *combination, __nv_bfloat16 *output) {\n" +
    "  const int32_t live_rows = counts[3];\n" +
    "  if (live_rows < 0 || live_rows > " +
    maximum_rows.to_string() +
    ") return;\n" +
    "  const int32_t row = (int32_t)blockIdx.x;\n" +
    "  if (row >= live_rows) return;\n" +
    "  const unsigned long long branch_base = (unsigned long long)row * " +
    hidden +
    "ULL;\n" +
    "  const unsigned long long state_base = (unsigned long long)row * " +
    flattened +
    "ULL;\n" +
    "  const unsigned long long vector_base = (unsigned long long)row * " +
    width +
    "ULL;\n" +
    "  const unsigned long long matrix_base = (unsigned long long)row * " +
    width +
    "ULL * " +
    width +
    "ULL;\n" +
    "  for (int32_t linear = (int32_t)threadIdx.x; linear < " +
    flattened +
    "; linear += (int32_t)blockDim.x) { const int32_t destination = linear / " +
    hidden +
    "; const int32_t column = linear - destination * " +
    hidden +
    "; const float branch_value = __bfloat162float(branch[branch_base + (unsigned long long)column]); const float post_value = post[vector_base + (unsigned long long)destination]; if (!isfinite(branch_value) || !isfinite(post_value)) continue; float value = __fmul_rn(post_value, branch_value); bool valid = isfinite(value); for (int32_t source = 0; source < " +
    width +
    "; ++source) { const float control = combination[matrix_base + (unsigned long long)destination * " +
    width +
    "ULL + (unsigned long long)source]; const float residual_value = __bfloat162float(residual[state_base + (unsigned long long)source * " +
    hidden +
    "ULL + (unsigned long long)column]); if (!isfinite(control) || !isfinite(residual_value)) { valid = false; break; } value = __fadd_rn(value, __fmul_rn(control, residual_value)); } if (valid && isfinite(value)) output[state_base + (unsigned long long)linear] = __float2bfloat16_rn(value); }\n" +
    "}\n",
  )
  {
    operation: PostCombineV1,
    numerical_contract: OrderedF32PostCombineBf16RoundV1,
    stream_width,
    hidden_width,
    maximum_rows,
    function_symbol,
    source,
  }
}
