///|
fn projection_cuda_prelude() -> String {
  "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n"
}

///|
fn projection_tensor_core_prelude() -> String {
  "#include <cuda_bf16.h>\n#include <mma.h>\n#include <math.h>\n#include <stdint.h>\n"
}

///|
fn projection_tensor_core_eligible(parameters : ProjectionParameters) -> Bool {
  parameters.family != GatedMlp &&
  parameters.input_width % 16 == 0 &&
  parameters.output_width % 16 == 0 &&
  (
    parameters.family != QkvProjection ||
    (
      parameters.query_width > 0 &&
      parameters.query_width % 16 == 0 &&
      parameters.intermediate_width > 0 &&
      parameters.intermediate_width % 16 == 0
    )
  )
}

///|
fn projection_tensor_core_mlp_eligible(
  parameters : ProjectionParameters,
) -> Bool {
  parameters.family == GatedMlp &&
  parameters.input_width % 256 == 0 &&
  parameters.output_width % 256 == 0 &&
  parameters.intermediate_width % 256 == 0 &&
  parameters.intermediate_width <= 3840
}

///|
fn projection_decode_workspace_bytes(
  parameters : ProjectionParameters,
) -> Int64 {
  if projection_tensor_core_mlp_eligible(parameters) {
    parameters.intermediate_width.to_int64() * 2L
  } else {
    0L
  }
}

///|
fn projection_workspace_bytes(
  parameters : ProjectionParameters,
  profile : @launch_contract.PagedKernelProfile,
) -> Int64 raise ProjectionAotError {
  let row_bytes = projection_decode_workspace_bytes(parameters)
  if row_bytes == 0L {
    0L
  } else {
    projection_checked_mul(row_bytes, profile.max_query_tokens().to_int64())
  }
}

///|
fn projection_count_guard(maximum_tokens : Int) -> String {
  "  const int token_count = counts[3];\n" +
  "  if (token_count <= 0 || token_count > " +
  maximum_tokens.to_string() +
  ") return;\n"
}

///|
fn projection_tensor_core_dense_source(
  symbol : String,
  parameters : ProjectionParameters,
  maximum_tokens : Int,
) -> Bytes {
  @utf8.encode(
    projection_tensor_core_prelude() +
    "extern \"C\" __global__ void " +
    symbol +
    "(const int32_t *counts, const __nv_bfloat16 *input, " +
    "const __nv_bfloat16 *weight, __nv_bfloat16 *output) {\n" +
    projection_count_guard(maximum_tokens) +
    "  const int warp = (int)threadIdx.x >> 5;\n" +
    "  const int lane = (int)threadIdx.x & 31;\n" +
    (if parameters.family != LanguageModelHead {
      "  if (token_count == 1) {\n" +
      "    const int output_column = (int)blockIdx.x * 8 + warp; if (output_column >= " +
      parameters.output_width.to_string() +
      ") return;\n" +
      "    const unsigned long long weight_base = (unsigned long long)output_column * " +
      parameters.input_width.to_string() +
      "ULL; float value = 0.0f;\n" +
      "    for (int inner = lane; inner < " +
      parameters.input_width.to_string() +
      "; inner += 32) value = __fadd_rn(value, __fmul_rn(__bfloat162float(input[inner]), __bfloat162float(weight[weight_base + inner])));\n" +
      "    for (int delta = 16; delta > 0; delta >>= 1) value = __fadd_rn(value, __shfl_down_sync(0xffffffffU, value, delta));\n" +
      "    if (lane == 0) output[output_column] = __float2bfloat16_rn(value);\n" +
      "    return;\n  }\n" +
      "  const int group_count = " +
      ((parameters.output_width + 127) / 128).to_string() +
      ";\n" +
      "  const int group_index = (int)blockIdx.x % group_count;\n" +
      "  const int row_begin = ((int)blockIdx.x / group_count) * 16;\n" +
      "  const int column = (group_index * 8 + warp) * 16;\n" +
      "  if (row_begin >= token_count || column >= " +
      parameters.output_width.to_string() +
      ") return;\n"
    } else {
      "  const int column = ((int)blockIdx.x * 8 + warp) * 16;\n" +
      "  if (column >= " +
      parameters.output_width.to_string() +
      ") return;\n"
    }) +
    (if parameters.family == LanguageModelHead {
      "  if (counts[2] == 1) {\n" +
      "    const unsigned long long live_row = (unsigned long long)(token_count - 1);\n" +
      "    const unsigned long long input_base = live_row * " +
      parameters.input_width.to_string() +
      "ULL;\n" +
      "    const unsigned long long output_base = live_row * " +
      parameters.output_width.to_string() +
      "ULL;\n" +
      "    #pragma unroll\n" +
      "    for (int item = 0; item < 16; ++item) {\n" +
      "      const int output_column = column + item; if (output_column >= " +
      parameters.output_width.to_string() +
      ") break;\n" +
      "      const unsigned long long weight_base = (unsigned long long)output_column * " +
      parameters.input_width.to_string() +
      "ULL; float value = 0.0f;\n" +
      "      for (int inner = lane; inner < " +
      parameters.input_width.to_string() +
      "; inner += 32) value = __fadd_rn(value, __fmul_rn(__bfloat162float(input[input_base + inner]), __bfloat162float(weight[weight_base + inner])));\n" +
      "      for (int delta = 16; delta > 0; delta >>= 1) value = __fadd_rn(value, __shfl_down_sync(0xffffffffU, value, delta));\n" +
      "      if (lane == 0) output[output_base + output_column] = __float2bfloat16_rn(value);\n" +
      "    } return;\n" +
      "  }\n"
    } else {
      ""
    }) +
    "  for (int row = " +
    (if parameters.family == LanguageModelHead { "0" } else { "row_begin" }) +
    "; row < token_count; row += " +
    (if parameters.family == LanguageModelHead {
      "16"
    } else {
      maximum_tokens.to_string()
    }) +
    ") {\n" +
    "  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> a;\n" +
    "  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::col_major> b;\n" +
    "  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c;\n" +
    "  nvcuda::wmma::fill_fragment(c, 0.0f);\n" +
    "  for (int inner = 0; inner < " +
    parameters.input_width.to_string() +
    "; inner += 16) {\n" +
    "    nvcuda::wmma::load_matrix_sync(a, input + (unsigned long long)row * " +
    parameters.input_width.to_string() +
    "ULL + inner, " +
    parameters.input_width.to_string() +
    ");\n" +
    "    nvcuda::wmma::load_matrix_sync(b, weight + (unsigned long long)column * " +
    parameters.input_width.to_string() +
    "ULL + inner, " +
    parameters.input_width.to_string() +
    ");\n" +
    "    nvcuda::wmma::mma_sync(c, a, b, c);\n" +
    "  }\n" +
    "  extern __shared__ float tile_output[];\n" +
    "  float *warp_output = tile_output + warp * 256;\n" +
    "  nvcuda::wmma::store_matrix_sync(warp_output, c, 16, nvcuda::wmma::mem_row_major);\n" +
    "  __syncwarp();\n" +
    "  for (int element = lane; element < 256; element += 32) {\n" +
    "    const int local_row = element >> 4;\n" +
    "    const int local_column = element & 15;\n" +
    "    if (row + local_row < token_count && column + local_column < " +
    parameters.output_width.to_string() +
    ") output[(unsigned long long)(row + local_row) * " +
    parameters.output_width.to_string() +
    "ULL + column + local_column] = __float2bfloat16_rn(warp_output[element]);\n" +
    "  }\n  }\n}\n",
  )
}

///|
fn projection_tensor_core_qkv_source(
  symbol : String,
  parameters : ProjectionParameters,
  maximum_tokens : Int,
) -> Bytes {
  @utf8.encode(
    projection_tensor_core_prelude() +
    "extern \"C\" __global__ void " +
    symbol +
    "(const int32_t *counts, const __nv_bfloat16 *input, " +
    "const __nv_bfloat16 *query_weight, const __nv_bfloat16 *key_weight, " +
    "const __nv_bfloat16 *value_weight, __nv_bfloat16 *output) {\n" +
    projection_count_guard(maximum_tokens) +
    "  const int warp = (int)threadIdx.x >> 5;\n" +
    "  const int lane = (int)threadIdx.x & 31;\n" +
    "  if (token_count == 1) {\n" +
    "    const int output_column = (int)blockIdx.x * 8 + warp; if (output_column >= " +
    parameters.output_width.to_string() +
    ") return;\n" +
    "    const __nv_bfloat16 *selected_weight = query_weight;\n" +
    "    int selected_column = output_column;\n" +
    "    if (output_column >= " +
    parameters.query_width.to_string() +
    ") { selected_column -= " +
    parameters.query_width.to_string() +
    "; if (selected_column >= " +
    parameters.intermediate_width.to_string() +
    ") { selected_column -= " +
    parameters.intermediate_width.to_string() +
    "; selected_weight = value_weight; } else selected_weight = key_weight; }\n" +
    "    const unsigned long long weight_base = (unsigned long long)selected_column * " +
    parameters.input_width.to_string() +
    "ULL; float value = 0.0f;\n" +
    "    for (int inner = lane; inner < " +
    parameters.input_width.to_string() +
    "; inner += 32) value = __fadd_rn(value, __fmul_rn(__bfloat162float(input[inner]), __bfloat162float(selected_weight[weight_base + inner])));\n" +
    "    for (int delta = 16; delta > 0; delta >>= 1) value = __fadd_rn(value, __shfl_down_sync(0xffffffffU, value, delta));\n" +
    "    if (lane == 0) output[output_column] = __float2bfloat16_rn(value);\n" +
    "    return;\n  }\n" +
    "  const int group_count = " +
    ((parameters.output_width + 127) / 128).to_string() +
    ";\n" +
    "  const int group_index = (int)blockIdx.x % group_count;\n" +
    "  const int row = ((int)blockIdx.x / group_count) * 16;\n" +
    "  const int column = (group_index * 8 + warp) * 16;\n" +
    "  if (row >= token_count || column >= " +
    parameters.output_width.to_string() +
    ") return;\n" +
    "  const __nv_bfloat16 *selected_weight = query_weight;\n" +
    "  int selected_column = column;\n" +
    "  if (column >= " +
    parameters.query_width.to_string() +
    ") { selected_column -= " +
    parameters.query_width.to_string() +
    "; if (selected_column >= " +
    parameters.intermediate_width.to_string() +
    ") { selected_column -= " +
    parameters.intermediate_width.to_string() +
    "; selected_weight = value_weight; } else selected_weight = key_weight; }\n" +
    "  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> a;\n" +
    "  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::col_major> b;\n" +
    "  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c;\n" +
    "  nvcuda::wmma::fill_fragment(c, 0.0f);\n" +
    "  for (int inner = 0; inner < " +
    parameters.input_width.to_string() +
    "; inner += 16) {\n" +
    "    nvcuda::wmma::load_matrix_sync(a, input + (unsigned long long)row * " +
    parameters.input_width.to_string() +
    "ULL + inner, " +
    parameters.input_width.to_string() +
    ");\n" +
    "    nvcuda::wmma::load_matrix_sync(b, selected_weight + (unsigned long long)selected_column * " +
    parameters.input_width.to_string() +
    "ULL + inner, " +
    parameters.input_width.to_string() +
    ");\n" +
    "    nvcuda::wmma::mma_sync(c, a, b, c);\n" +
    "  }\n" +
    "  extern __shared__ float tile_output[];\n" +
    "  float *warp_output = tile_output + warp * 256;\n" +
    "  nvcuda::wmma::store_matrix_sync(warp_output, c, 16, nvcuda::wmma::mem_row_major);\n" +
    "  __syncwarp();\n" +
    "  for (int element = lane; element < 256; element += 32) {\n" +
    "    const int local_row = element >> 4; const int local_column = element & 15;\n" +
    "    if (row + local_row < token_count && column + local_column < " +
    parameters.output_width.to_string() +
    ") output[(unsigned long long)(row + local_row) * " +
    parameters.output_width.to_string() +
    "ULL + column + local_column] = __float2bfloat16_rn(warp_output[element]);\n" +
    "  }\n}\n",
  )
}

///|
fn projection_dense_source(
  symbol : String,
  parameters : ProjectionParameters,
  maximum_tokens : Int,
) -> Bytes {
  @utf8.encode(
    projection_cuda_prelude() +
    "extern \"C\" __global__ void " +
    symbol +
    "(const int32_t *counts, const __nv_bfloat16 *input, " +
    "const __nv_bfloat16 *weight, __nv_bfloat16 *output) {\n" +
    projection_count_guard(maximum_tokens) +
    "  const unsigned long long index = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;\n" +
    "  const unsigned long long live = (unsigned long long)token_count * " +
    parameters.output_width.to_string() +
    "ULL;\n" +
    "  if (index >= live) return;\n" +
    "  const unsigned long long row = index / " +
    parameters.output_width.to_string() +
    "ULL;\n" +
    "  const unsigned long long column = index % " +
    parameters.output_width.to_string() +
    "ULL;\n" +
    "  const unsigned long long input_base = row * " +
    parameters.input_width.to_string() +
    "ULL;\n" +
    "  const unsigned long long weight_base = column * " +
    parameters.input_width.to_string() +
    "ULL;\n" +
    "  float accumulator = 0.0f;\n" +
    "  for (int inner = 0; inner < " +
    parameters.input_width.to_string() +
    "; ++inner) {\n" +
    "    const float product = __fmul_rn(__bfloat162float(input[input_base + inner]), " +
    "__bfloat162float(weight[weight_base + inner]));\n" +
    "    accumulator = __fadd_rn(accumulator, product);\n" +
    "  }\n" +
    "  output[index] = __float2bfloat16_rn(accumulator);\n}\n",
  )
}

///|
fn projection_qkv_source(
  symbol : String,
  parameters : ProjectionParameters,
  maximum_tokens : Int,
) -> Bytes {
  @utf8.encode(
    projection_cuda_prelude() +
    "extern \"C\" __global__ void " +
    symbol +
    "(const int32_t *counts, const __nv_bfloat16 *input, " +
    "const __nv_bfloat16 *query_weight, const __nv_bfloat16 *key_weight, " +
    "const __nv_bfloat16 *value_weight, __nv_bfloat16 *output) {\n" +
    projection_count_guard(maximum_tokens) +
    "  const unsigned long long index = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;\n" +
    "  const unsigned long long live = (unsigned long long)token_count * " +
    parameters.output_width.to_string() +
    "ULL;\n" +
    "  if (index >= live) return;\n" +
    "  const unsigned long long row = index / " +
    parameters.output_width.to_string() +
    "ULL;\n" +
    "  const int column = (int)(index % " +
    parameters.output_width.to_string() +
    "ULL);\n" +
    "  const __nv_bfloat16 *weight = query_weight;\n" +
    "  int local_column = column;\n" +
    "  if (column >= " +
    parameters.query_width.to_string() +
    ") {\n" +
    "    local_column = column - " +
    parameters.query_width.to_string() +
    ";\n" +
    "    if (local_column >= " +
    parameters.intermediate_width.to_string() +
    ") { weight = value_weight; local_column -= " +
    parameters.intermediate_width.to_string() +
    "; }\n" +
    "    else { weight = key_weight; }\n" +
    "  }\n" +
    "  const unsigned long long input_base = row * " +
    parameters.input_width.to_string() +
    "ULL;\n" +
    "  const unsigned long long weight_base = (unsigned long long)local_column * " +
    parameters.input_width.to_string() +
    "ULL;\n" +
    "  float accumulator = 0.0f;\n" +
    "  for (int inner = 0; inner < " +
    parameters.input_width.to_string() +
    "; ++inner) {\n" +
    "    const float product = __fmul_rn(__bfloat162float(input[input_base + inner]), " +
    "__bfloat162float(weight[weight_base + inner]));\n" +
    "    accumulator = __fadd_rn(accumulator, product);\n" +
    "  }\n" +
    "  output[index] = __float2bfloat16_rn(accumulator);\n}\n",
  )
}
