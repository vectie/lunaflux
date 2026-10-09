///|
fn valid_projection_symbol(symbol : StringView) -> Bool {
  if symbol.is_empty() || symbol.length() > 128 {
    return false
  }
  symbol[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
/// Renders a deterministic group-local BF16 projection. One CUDA block owns
/// one `(row,group)` pair, and threads cover output columns in increasing
/// strides. Each result uses ordered F32 multiply/add and one BF16-RNE round.
pub fn grouped_projection_bf16_source(
  group_count~ : Int,
  input_width~ : Int,
  output_width~ : Int,
  maximum_rows~ : Int,
  function_symbol~ : String,
) -> GroupedDecoderProjectionCudaSource raise DecoderProjectionCudaSourceError {
  if group_count <= 0 || group_count > 1024 {
    raise InvalidInputWidth
  }
  if input_width <= 0 {
    raise InvalidInputWidth
  }
  if output_width <= 0 {
    raise InvalidOutputWidth
  }
  if maximum_rows <= 0 || maximum_rows > 1048576 {
    raise InvalidMaximumRows
  }
  if !valid_projection_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const __nv_bfloat16 *input, const __nv_bfloat16 *weight, __nv_bfloat16 *output) {\n" +
    "  const int rows = counts[3];\n  if (rows <= 0 || rows > " +
    maximum_rows.to_string() +
    ") return;\n" +
    "  const int row = (int)blockIdx.x;\n  const int group = (int)blockIdx.y;\n" +
    "  if (row >= rows || group >= " +
    group_count.to_string() +
    ") return;\n" +
    "  for (int column = (int)threadIdx.x; column < " +
    output_width.to_string() +
    "; column += 256) {\n" +
    "    const unsigned long long input_base = ((unsigned long long)row * " +
    group_count.to_string() +
    "ULL + group) * " +
    input_width.to_string() +
    "ULL;\n" +
    "    const unsigned long long weight_base = ((unsigned long long)group * " +
    output_width.to_string() +
    "ULL + column) * " +
    input_width.to_string() +
    "ULL;\n" +
    "    float accumulator = 0.0f;\n" +
    "    for (int inner = 0; inner < " +
    input_width.to_string() +
    "; ++inner) {\n" +
    "      const float product = __fmul_rn(__bfloat162float(input[input_base + inner]), __bfloat162float(weight[weight_base + inner]));\n" +
    "      accumulator = __fadd_rn(accumulator, product);\n    }\n" +
    "    const unsigned long long out = ((unsigned long long)row * " +
    group_count.to_string() +
    "ULL + group) * " +
    output_width.to_string() +
    "ULL + column;\n" +
    "    output[out] = __float2bfloat16_rn(accumulator);\n  }\n}\n",
  )
  {
    operation: GroupedDenseProjectionBf16V1,
    numerical_contract: OrderedFloat32AccumulationBf16RoundV1,
    group_count,
    input_width,
    output_width,
    maximum_rows,
    function_symbol,
    source,
  }
}

///|
/// Renders one correctness-grade BF16 dense projection. Weights are row-major
/// `[output_width, input_width]`; there is deliberately no bias operand.
pub fn dense_projection_bf16_source(
  input_width~ : Int,
  output_width~ : Int,
  maximum_rows~ : Int,
  function_symbol~ : String,
) -> DecoderProjectionCudaSource raise DecoderProjectionCudaSourceError {
  if input_width <= 0 {
    raise InvalidInputWidth
  }
  if output_width <= 0 {
    raise InvalidOutputWidth
  }
  if maximum_rows <= 0 || maximum_rows > 1048576 {
    raise InvalidMaximumRows
  }
  if !valid_projection_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const __nv_bfloat16 *input, const __nv_bfloat16 *weight, __nv_bfloat16 *output) {\n" +
    "  const int row_count = counts[3];\n" +
    "  if (row_count <= 0 || row_count > " +
    maximum_rows.to_string() +
    ") return;\n" +
    "  const unsigned long long index = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;\n" +
    "  const unsigned long long live = (unsigned long long)row_count * " +
    output_width.to_string() +
    "ULL;\n" +
    "  if (index >= live) return;\n" +
    "  const unsigned long long row = index / " +
    output_width.to_string() +
    "ULL;\n" +
    "  const unsigned long long column = index % " +
    output_width.to_string() +
    "ULL;\n" +
    "  const unsigned long long input_base = row * " +
    input_width.to_string() +
    "ULL;\n" +
    "  const unsigned long long weight_base = column * " +
    input_width.to_string() +
    "ULL;\n" +
    "  float accumulator = 0.0f;\n" +
    "  for (int inner = 0; inner < " +
    input_width.to_string() +
    "; ++inner) {\n" +
    "    const float product = __fmul_rn(__bfloat162float(input[input_base + inner]), __bfloat162float(weight[weight_base + inner]));\n" +
    "    accumulator = __fadd_rn(accumulator, product);\n" +
    "  }\n" +
    "  output[index] = __float2bfloat16_rn(accumulator);\n}\n",
  )
  {
    operation: DenseProjectionBf16,
    numerical_contract: OrderedFloat32AccumulationBf16RoundV1,
    input_width,
    output_width,
    maximum_rows,
    function_symbol,
    source,
  }
}
