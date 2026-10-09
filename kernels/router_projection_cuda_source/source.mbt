///|
fn valid_router_projection_symbol(value : StringView) -> Bool {
  !value.is_empty() &&
  value.length() <= 128 &&
  value[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  value.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
pub fn f32_input_bf16_weight_projection_source(
  input_width~ : Int,
  output_width~ : Int,
  maximum_rows~ : Int,
  function_symbol~ : String,
) -> RouterProjectionCudaSource raise RouterProjectionCudaSourceError {
  if input_width <= 0 {
    raise InvalidInputWidth
  }
  if output_width <= 0 {
    raise InvalidOutputWidth
  }
  if maximum_rows <= 0 || maximum_rows > 1048576 {
    raise InvalidMaximumRows
  }
  if !valid_router_projection_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let output_cells = maximum_rows.to_int64() * output_width.to_int64()
  if output_cells <= 0L || output_cells > 9223372036854775552L {
    raise DimensionOverflow
  }
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const float *input, const __nv_bfloat16 *weight, float *output) {\n" +
    "  const int row_count = counts[3]; if (row_count <= 0 || row_count > " +
    maximum_rows.to_string() +
    ") return;\n" +
    "  const unsigned long long index = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x; const unsigned long long live = (unsigned long long)row_count * " +
    output_width.to_string() +
    "ULL; if (index >= live) return;\n" +
    "  const unsigned long long row = index / " +
    output_width.to_string() +
    "ULL; const unsigned long long column = index % " +
    output_width.to_string() +
    "ULL; const unsigned long long input_base = row * " +
    input_width.to_string() +
    "ULL; const unsigned long long weight_base = column * " +
    input_width.to_string() +
    "ULL; float accumulator = 0.0f;\n" +
    "  for (int inner = 0; inner < " +
    input_width.to_string() +
    "; ++inner) { const float product = __fmul_rn(input[input_base + inner], __bfloat162float(weight[weight_base + inner])); accumulator = __fadd_rn(accumulator, product); }\n" +
    "  output[index] = accumulator;\n}\n",
  )
  {
    operation: Float32InputBf16WeightFloat32OutputProjection,
    numerical_contract: OrderedFloat32AccumulationNoBiasV1,
    input_width,
    output_width,
    maximum_rows,
    function_symbol,
    source,
  }
}
