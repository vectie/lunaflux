///|
fn cuda_prelude() -> String {
  "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n"
}

///|
fn embedding_source(symbol : String, parameters : PointwiseParameters) -> Bytes {
  @utf8.encode(
    cuda_prelude() +
    "extern \"C\" __global__ void " +
    symbol +
    "(const int32_t *counts, const int32_t *token_ids, " +
    "const __nv_bfloat16 *table, __nv_bfloat16 *output) {\n" +
    "  const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;\n" +
    "  const int live_tokens = counts[3];\n  if (live_tokens <= 0) return;\n" +
    "  const unsigned long long live = (unsigned long long)live_tokens * " +
    parameters.width.to_string() +
    "ULL;\n" +
    "  if (i < live) {\n" +
    "    const int32_t token = token_ids[i / " +
    parameters.width.to_string() +
    "ULL];\n" +
    "    if (token >= 0 && token < " +
    parameters.vocabulary_size.to_string() +
    ") {\n" +
    "      output[i] = table[(unsigned long long)token * " +
    parameters.width.to_string() +
    "ULL + i % " +
    parameters.width.to_string() +
    "ULL];\n" +
    "    } else { output[i] = __float2bfloat16_rn(0.0f); }\n  }\n}\n",
  )
}

///|
fn residual_source(symbol : String, parameters : PointwiseParameters) -> Bytes {
  @utf8.encode(
    cuda_prelude() +
    "extern \"C\" __global__ void " +
    symbol +
    "(const int32_t *counts, const __nv_bfloat16 *left, " +
    "const __nv_bfloat16 *right, __nv_bfloat16 *output) {\n" +
    "  const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;\n" +
    "  const int live_tokens = counts[3];\n  if (live_tokens <= 0) return;\n" +
    "  const unsigned long long live = (unsigned long long)live_tokens * " +
    parameters.width.to_string() +
    "ULL;\n" +
    "  if (i < live) {\n" +
    "    const float sum = __bfloat162float(left[i]) + __bfloat162float(right[i]);\n" +
    "    output[i] = __float2bfloat16_rn(sum);\n  }\n}\n",
  )
}

///|
fn rms_norm_source(symbol : String, parameters : PointwiseParameters) -> Bytes {
  @utf8.encode(
    cuda_prelude() +
    "extern \"C\" __global__ void " +
    symbol +
    "(const int32_t *counts, const __nv_bfloat16 *input, " +
    "const __nv_bfloat16 *weight, __nv_bfloat16 *output) {\n" +
    "  const int row = (int)blockIdx.x;\n  if (counts[3] <= 0 || row >= counts[3]) return;\n" +
    "  __shared__ float partial[256];\n  float sum = 0.0f;\n" +
    "  const unsigned long long base = (unsigned long long)row * " +
    parameters.width.to_string() +
    "ULL;\n" +
    "  for (int column = threadIdx.x; column < " +
    parameters.width.to_string() +
    "; column += 256) {\n" +
    "    const float value = __bfloat162float(input[base + column]);\n" +
    "    sum = sum + value * value;\n  }\n" +
    "  partial[threadIdx.x] = sum;\n  __syncthreads();\n" +
    "  for (int stride = 128; stride > 0; stride >>= 1) {\n" +
    "    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];\n" +
    "    __syncthreads();\n  }\n" +
    "  const float inverse = rsqrtf(partial[0] / " +
    parameters.width.to_string() +
    ".0f + " +
    canonical_cuda_float_literal(parameters.epsilon) +
    ");\n" +
    "  for (int column = threadIdx.x; column < " +
    parameters.width.to_string() +
    "; column += 256) {\n" +
    "    const float value = __bfloat162float(input[base + column]);\n" +
    "    const float scale = __bfloat162float(weight[column]);\n" +
    "    output[base + column] = __float2bfloat16_rn(value * inverse * scale);\n" +
    "  }\n}\n",
  )
}

///|
fn rotary_source(symbol : String, parameters : PointwiseParameters) -> Bytes {
  let rotary_width = (parameters.query_heads + parameters.kv_heads) *
    parameters.head_dimension
  let half = parameters.head_dimension / 2
  @utf8.encode(
    cuda_prelude() +
    "extern \"C\" __global__ void " +
    symbol +
    "(const int32_t *counts, const int32_t *positions, " +
    "const __nv_bfloat16 *input, __nv_bfloat16 *output) {\n" +
    "  const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;\n" +
    "  const int live_tokens = counts[3];\n  if (live_tokens <= 0) return;\n" +
    "  const unsigned long long live = (unsigned long long)live_tokens * " +
    parameters.width.to_string() +
    "ULL;\n  if (i >= live) return;\n" +
    "  const int column = (int)(i % " +
    parameters.width.to_string() +
    "ULL);\n" +
    "  if (column >= " +
    rotary_width.to_string() +
    ") { output[i] = input[i]; return; }\n" +
    "  const int lane = column % " +
    parameters.head_dimension.to_string() +
    ";\n" +
    "  const int pair = lane % " +
    half.to_string() +
    ";\n" +
    "  const int head_base = column - lane;\n" +
    "  const unsigned long long row_base = i - (unsigned long long)column;\n" +
    "  const float exponent = (float)(pair * 2) / " +
    parameters.head_dimension.to_string() +
    ".0f;\n" +
    "  const float angle = (float)positions[i / " +
    parameters.width.to_string() +
    "ULL] / powf(" +
    canonical_cuda_float_literal(parameters.rope_theta) +
    ", exponent);\n" +
    "  const float cosine = cosf(angle);\n  const float sine = sinf(angle);\n" +
    "  const float first = __bfloat162float(input[row_base + head_base + pair]);\n" +
    "  const float second = __bfloat162float(input[row_base + head_base + pair + " +
    half.to_string() +
    "]);\n" +
    "  const float value = lane < " +
    half.to_string() +
    " ? first * cosine - second * sine : second * cosine + first * sine;\n" +
    "  output[i] = __float2bfloat16_rn(value);\n}\n",
  )
}

///|
fn pointwise_source(
  symbol : String,
  parameters : PointwiseParameters,
) -> Bytes raise PointwiseAotError {
  match parameters.family {
    EmbeddingLookup => embedding_source(symbol, parameters)
    RmsNorm => rms_norm_source(symbol, parameters)
    PositionedRotary => rotary_source(symbol, parameters)
    ResidualAdd => residual_source(symbol, parameters)
    KvCacheWrite => raise Unsupported(Operation)
  }
}
