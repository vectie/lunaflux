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
    "  const int live_tokens = counts[3];\n  if (live_tokens <= 0) return;\n" +
    "  const unsigned long long live = (unsigned long long)live_tokens * " +
    parameters.width.to_string() +
    "ULL;\n" +
    "  for (unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x; i < live; i += (unsigned long long)gridDim.x * 256ULL) {\n" +
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
    "  const int live_tokens = counts[3];\n  if (live_tokens <= 0) return;\n" +
    "  const unsigned long long live = (unsigned long long)live_tokens * " +
    parameters.width.to_string() +
    "ULL;\n" +
    "  for (unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x; i < live; i += (unsigned long long)gridDim.x * 256ULL) {\n" +
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
    "  if (counts[3] <= 0) return;\n" +
    "  if (blockDim.x != 256) return;\n" +
    "  const int lane = (int)threadIdx.x & 31;\n" +
    "  const int warp = (int)threadIdx.x >> 5;\n" +
    "  __shared__ float warp_sums[8];\n" +
    "  for (int row = (int)blockIdx.x; row < counts[3]; row += (int)gridDim.x) {\n" +
    "  float sum = 0.0f;\n" +
    "  const unsigned long long base = (unsigned long long)row * " +
    parameters.width.to_string() +
    "ULL;\n" +
    "  for (int column = (int)threadIdx.x; column < " +
    parameters.width.to_string() +
    "; column += 256) {\n" +
    "    const float value = __bfloat162float(input[base + column]);\n" +
    "    sum = sum + value * value;\n  }\n" +
    "  for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffffU, sum, offset);\n" +
    "  if (lane == 0) warp_sums[warp] = sum;\n" +
    "  __syncthreads();\n" +
    "  float total = lane < 8 ? warp_sums[lane] : 0.0f;\n" +
    "  if (warp == 0) {\n" +
    "    for (int offset = 16; offset > 0; offset >>= 1) total += __shfl_down_sync(0xffffffffU, total, offset);\n" +
    "    if (lane == 0) warp_sums[0] = total;\n" +
    "  }\n" +
    "  __syncthreads();\n" +
    "  const float inverse = rsqrtf(warp_sums[0] / " +
    parameters.width.to_string() +
    ".0f + " +
    canonical_cuda_float_literal(parameters.epsilon) +
    ");\n" +
    "  for (int column = (int)threadIdx.x; column < " +
    parameters.width.to_string() +
    "; column += 256) {\n" +
    "    const float value = __bfloat162float(input[base + column]);\n" +
    "    const float scale = __bfloat162float(weight[column]);\n" +
    "    output[base + column] = __float2bfloat16_rn(value * inverse * scale);\n" +
    "  }\n  }\n}\n",
  )
}

///|
fn qk_rms_norm_source(
  symbol : String,
  parameters : PointwiseParameters,
) -> Bytes {
  let packed_heads = parameters.query_heads + 2 * parameters.kv_heads
  @utf8.encode(
    cuda_prelude() +
    "extern \"C\" __global__ void " +
    symbol +
    "(const int32_t *counts, const __nv_bfloat16 *input, " +
    "const __nv_bfloat16 *q_norm, const __nv_bfloat16 *k_norm, " +
    "__nv_bfloat16 *output) {\n" +
    "  const int warp = (int)threadIdx.x >> 5;\n" +
    "  const int lane = (int)threadIdx.x & 31;\n" +
    "  for (int work = (int)blockIdx.x * 8 + warp; work < counts[3] * " +
    packed_heads.to_string() +
    "; work += (int)gridDim.x * 8) {\n" +
    "  const int packed_head = work % " +
    packed_heads.to_string() +
    ";\n" +
    "  const int row = work / " +
    packed_heads.to_string() +
    ";\n" +
    "  if (counts[3] <= 0 || row >= counts[3]) continue;\n" +
    "  const unsigned long long base = ((unsigned long long)row * " +
    packed_heads.to_string() +
    "ULL + (unsigned long long)packed_head) * " +
    parameters.head_dimension.to_string() +
    "ULL;\n" +
    "  if (packed_head >= " +
    (parameters.query_heads + parameters.kv_heads).to_string() +
    ") {\n" +
    "    for (int column = lane; column < " +
    parameters.head_dimension.to_string() +
    "; column += 32) output[base + column] = input[base + column];\n" +
    "    continue;\n  }\n" +
    "  float sum = 0.0f;\n" +
    "  for (int column = lane; column < " +
    parameters.head_dimension.to_string() +
    "; column += 32) {\n" +
    "    const float value = __bfloat162float(input[base + column]);\n" +
    "    sum = sum + value * value;\n  }\n" +
    "  for (int offset = 16; offset > 0; offset >>= 1)\n" +
    "    sum += __shfl_down_sync(0xffffffffU, sum, offset);\n" +
    "  const float inverse = rsqrtf(__shfl_sync(0xffffffffU, sum, 0) / " +
    parameters.head_dimension.to_string() +
    ".0f + " +
    canonical_cuda_float_literal(parameters.epsilon) +
    ");\n" +
    "  const __nv_bfloat16 *weight = packed_head < " +
    parameters.query_heads.to_string() +
    " ? q_norm : k_norm;\n" +
    "  for (int column = lane; column < " +
    parameters.head_dimension.to_string() +
    "; column += 32) {\n" +
    "    const float value = __bfloat162float(input[base + column]);\n" +
    "    const float scale = __bfloat162float(weight[column]);\n" +
    "    output[base + column] = __float2bfloat16_rn(value * inverse * scale);\n" +
    "  }\n  }\n}\n",
  )
}

///|
fn rotary_source(symbol : String, parameters : PointwiseParameters) -> Bytes {
  let rotary_heads = parameters.query_heads + parameters.kv_heads
  let rotary_width = rotary_heads * parameters.head_dimension
  let half = parameters.head_dimension / 2
  let passthrough_width = parameters.width - rotary_width
  let passthrough_groups = (passthrough_width + 31) / 32
  let work_per_token = half + passthrough_groups
  @utf8.encode(
    cuda_prelude() +
    "extern \"C\" __global__ void " +
    symbol +
    "(const int32_t *counts, const int32_t *positions, " +
    "const __nv_bfloat16 *input, __nv_bfloat16 *output) {\n" +
    "  const int live_tokens = counts[3];\n  if (live_tokens <= 0) return;\n" +
    "  if (blockDim.x != 32) return;\n" +
    "  const int lane = (int)threadIdx.x;\n" +
    "  const unsigned long long live = (unsigned long long)live_tokens * " +
    work_per_token.to_string() +
    "ULL;\n" +
    "  for (unsigned long long work = (unsigned long long)blockIdx.x; work < live; work += (unsigned long long)gridDim.x) {\n" +
    "  const unsigned long long row = work / " +
    work_per_token.to_string() +
    "ULL;\n" +
    "  const int item = (int)(work % " +
    work_per_token.to_string() +
    "ULL);\n" +
    "  const unsigned long long row_base = row * " +
    parameters.width.to_string() +
    "ULL;\n" +
    "  if (item >= " +
    half.to_string() +
    ") { const int column = " +
    rotary_width.to_string() +
    " + (item - " +
    half.to_string() +
    ") * 32 + lane; if (column < " +
    parameters.width.to_string() +
    ") output[row_base + column] = input[row_base + column]; continue; }\n" +
    "  const int pair = item;\n" +
    "  float cosine = 0.0f, sine = 0.0f;\n" +
    "  if (lane == 0) { const float exponent = (float)(pair * 2) / " +
    parameters.head_dimension.to_string() +
    ".0f;\n" +
    "    const float angle = (float)positions[row] / powf(" +
    canonical_cuda_float_literal(parameters.rope_theta) +
    ", exponent); cosine = cosf(angle); sine = sinf(angle); }\n" +
    "  cosine = __shfl_sync(0xffffffffU, cosine, 0); sine = __shfl_sync(0xffffffffU, sine, 0);\n" +
    "  if (lane < " +
    rotary_heads.to_string() +
    ") { const int head = lane;\n" +
    "    const unsigned long long first_offset = row_base + (unsigned long long)head * " +
    parameters.head_dimension.to_string() +
    "ULL + pair;\n" +
    "    const unsigned long long second_offset = first_offset + " +
    half.to_string() +
    "ULL;\n" +
    "    const float first = __bfloat162float(input[first_offset]);\n" +
    "    const float second = __bfloat162float(input[second_offset]);\n" +
    "    output[first_offset] = __float2bfloat16_rn(first * cosine - second * sine);\n" +
    "    output[second_offset] = __float2bfloat16_rn(second * cosine + first * sine);\n" +
    "  }\n  }\n}\n",
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
    QkRmsNorm => qk_rms_norm_source(symbol, parameters)
    PositionedRotary => rotary_source(symbol, parameters)
    ResidualAdd => residual_source(symbol, parameters)
    KvCacheWrite => raise Unsupported(Operation)
  }
}
