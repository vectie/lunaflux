///|
/// Renders one correctness-grade fixed-row F32 dense projection with bias.
/// Inputs are row-major `[rows,input_width]`, weights are row-major
/// `[output_width,input_width]`, bias is `[output_width]`, and output is
/// row-major F32 `[rows,output_width]`.
pub fn dense_projection_f32_bias_f32_source(
  rows~ : Int64,
  input_width~ : Int,
  output_width~ : Int,
  function_symbol~ : String,
) -> FixedRowsDecoderProjectionCudaSource raise DecoderProjectionCudaSourceError {
  if rows <= 0L || rows > 2147483647L {
    raise InvalidRowCount
  }
  if input_width <= 0 {
    raise InvalidInputWidth
  }
  if output_width <= 0 {
    raise InvalidOutputWidth
  }
  if !valid_projection_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let live = rows * output_width.to_int64()
  let source = @utf8.encode(
    "#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const float *input, const float *weight, const float *bias, float *output) {\n" +
    "  const unsigned long long index = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;\n" +
    "  if (index >= " +
    live.to_string() +
    "ULL) return;\n" +
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
    "  float accumulator = bias[column];\n" +
    "  for (int inner = 0; inner < " +
    input_width.to_string() +
    "; ++inner) {\n" +
    "    const float product = __fmul_rn(input[input_base + inner], weight[weight_base + inner]);\n" +
    "    accumulator = __fadd_rn(accumulator, product);\n" +
    "  }\n" +
    "  output[index] = accumulator;\n}\n",
  )
  {
    operation: DenseProjectionF32BiasF32,
    numerical_contract: OrderedFloat32AccumulationF32OutputV1,
    row_count: rows,
    input_width,
    output_width,
    function_symbol,
    source,
  }
}

///|
/// Renders one correctness-grade fixed-row F32 dense projection with bias and
/// one final BF16 round. Inputs are row-major `[rows,input_width]`, weights are
/// row-major `[output_width,input_width]`, bias is `[output_width]`, and output
/// is row-major BF16 `[rows,output_width]`.
pub fn dense_projection_f32_bias_bf16_source(
  rows~ : Int64,
  input_width~ : Int,
  output_width~ : Int,
  function_symbol~ : String,
) -> FixedRowsDecoderProjectionCudaSource raise DecoderProjectionCudaSourceError {
  if rows <= 0L || rows > 2147483647L {
    raise InvalidRowCount
  }
  if input_width <= 0 {
    raise InvalidInputWidth
  }
  if output_width <= 0 {
    raise InvalidOutputWidth
  }
  if !valid_projection_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let live = rows * output_width.to_int64()
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const float *input, const float *weight, const float *bias, __nv_bfloat16 *output) {\n" +
    "  const unsigned long long index = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;\n" +
    "  if (index >= " +
    live.to_string() +
    "ULL) return;\n" +
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
    "  float accumulator = bias[column];\n" +
    "  for (int inner = 0; inner < " +
    input_width.to_string() +
    "; ++inner) {\n" +
    "    const float product = __fmul_rn(input[input_base + inner], weight[weight_base + inner]);\n" +
    "    accumulator = __fadd_rn(accumulator, product);\n" +
    "  }\n" +
    "  output[index] = __float2bfloat16_rn(accumulator);\n}\n",
  )
  {
    operation: DenseProjectionF32BiasBf16,
    numerical_contract: OrderedFloat32BiasAccumulationBf16RoundV1,
    row_count: rows,
    input_width,
    output_width,
    function_symbol,
    source,
  }
}
