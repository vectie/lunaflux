///|
fn projection_cuda_prelude() -> String {
  "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n"
}

///|
fn projection_count_guard(maximum_tokens : Int) -> String {
  "  const int token_count = counts[3];\n" +
  "  if (token_count <= 0 || token_count > " +
  maximum_tokens.to_string() +
  ") return;\n"
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
    parameters.input_width.to_string() +
    ") {\n" +
    "    local_column = column - " +
    parameters.input_width.to_string() +
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
