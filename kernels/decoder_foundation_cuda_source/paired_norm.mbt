///|
fn paired_rms_norm_branch(
  stream : Int,
  width : Int,
  input_name : String,
  weight_name : String,
  output_name : String,
  epsilon_literal : String,
) -> String {
  "  if (stream == " +
  stream.to_string() +
  ") {\n" +
  "    const unsigned long long base = (unsigned long long)row * " +
  width.to_string() +
  "ULL;\n" +
  "    float sum = 0.0f;\n" +
  "    for (int column = (int)threadIdx.x; column < " +
  width.to_string() +
  "; column += 256) {\n" +
  "      const float value = __bfloat162float(" +
  input_name +
  "[base + (unsigned long long)column]);\n" +
  "      sum = __fadd_rn(sum, __fmul_rn(value, value));\n" +
  "    }\n" +
  "    partial[threadIdx.x] = sum;\n" +
  "    __syncthreads();\n" +
  "    for (int stride = 128; stride > 0; stride >>= 1) {\n" +
  "      if ((int)threadIdx.x < stride) partial[threadIdx.x] = __fadd_rn(partial[threadIdx.x], partial[threadIdx.x + stride]);\n" +
  "      __syncthreads();\n" +
  "    }\n" +
  "    const float inverse = rsqrtf(__fadd_rn(__fdiv_rn(partial[0], " +
  width.to_string() +
  ".0f), " +
  epsilon_literal +
  "));\n" +
  "    for (int column = (int)threadIdx.x; column < " +
  width.to_string() +
  "; column += 256) {\n" +
  "      const float value = __bfloat162float(" +
  input_name +
  "[base + (unsigned long long)column]);\n" +
  "      const float scale = __bfloat162float(" +
  weight_name +
  "[column]);\n" +
  "      " +
  output_name +
  "[base + (unsigned long long)column] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(value, inverse), scale));\n" +
  "    }\n" +
  "  }\n"
}

///|
/// Emits one two-stream row-wise RMSNorm kernel. The first and second BF16
/// streams have independent row widths and weights but share one positive
/// epsilon and the live row prefix in `counts[3]`. The launch geometry is
/// `[maximum_rows, 2, 1]` blocks of 256 threads with 1024 shared bytes.
pub fn paired_rms_norm_bf16_source(
  first_width~ : Int,
  second_width~ : Int,
  maximum_rows~ : Int,
  epsilon~ : Double,
  function_symbol~ : String,
) -> DecoderFoundationCudaSource raise DecoderFoundationCudaSourceError {
  if first_width <= 0 || second_width <= 0 {
    raise InvalidHiddenSize
  }
  if maximum_rows <= 0 || maximum_rows > 1048576 {
    raise InvalidRowCount
  }
  if epsilon.is_nan() || epsilon.is_inf() || epsilon <= 0.0 || epsilon > 1.0 {
    raise InvalidEpsilon
  }
  if !valid_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let epsilon_literal = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(
    epsilon,
  )
  let source = @utf8.encode(
    prelude() +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const __nv_bfloat16 *first_input, const __nv_bfloat16 *first_weight, __nv_bfloat16 *first_output, const __nv_bfloat16 *second_input, const __nv_bfloat16 *second_weight, __nv_bfloat16 *second_output) {\n" +
    "  const int32_t live_rows = counts[3];\n" +
    "  if (live_rows < 0 || live_rows > " +
    maximum_rows.to_string() +
    ") return;\n" +
    "  const int32_t row = (int32_t)blockIdx.x;\n" +
    "  const int32_t stream = (int32_t)blockIdx.y;\n" +
    "  if (row >= live_rows || stream < 0 || stream > 1) return;\n" +
    "  __shared__ float partial[256];\n" +
    paired_rms_norm_branch(
      0, first_width, "first_input", "first_weight", "first_output", epsilon_literal,
    ) +
    paired_rms_norm_branch(
      1, second_width, "second_input", "second_weight", "second_output", epsilon_literal,
    ) +
    "}\n",
  )
  {
    operation: PairedRmsNormBf16,
    numerical_contract: OrderedFloat32ReductionBf16RoundV1,
    function_symbol,
    source,
  }
}
