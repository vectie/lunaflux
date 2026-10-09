///|
/// Renders one correctness-grade fixed-row BF16 RMSNorm. Inputs are row-major
/// `[rows,hidden_size]`, weights are `[hidden_size]`, and output is row-major
/// BF16 `[rows,hidden_size]`. One 256-thread block owns each row.
pub fn rms_norm_bf16_fixed_rows_source(
  rows~ : Int64,
  hidden_size~ : Int,
  epsilon~ : Double,
  function_symbol~ : String,
) -> DecoderFoundationCudaSource raise DecoderFoundationCudaSourceError {
  if rows <= 0L || rows > 2147483647L {
    raise InvalidRowCount
  }
  if hidden_size <= 0 {
    raise InvalidHiddenSize
  }
  if epsilon.is_nan() || epsilon.is_inf() || epsilon <= 0.0 || epsilon > 1.0 {
    raise InvalidEpsilon
  }
  if !valid_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let source = @utf8.encode(
    prelude() +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const __nv_bfloat16 *input, const __nv_bfloat16 *weight, __nv_bfloat16 *output) {\n" +
    "  const unsigned long long row = (unsigned long long)blockIdx.x;\n" +
    "  if (row >= " +
    rows.to_string() +
    "ULL) return;\n" +
    "  __shared__ float partial[256];\n  float sum = 0.0f;\n" +
    "  const unsigned long long base = row * " +
    hidden_size.to_string() +
    "ULL;\n" +
    "  for (int column = threadIdx.x; column < " +
    hidden_size.to_string() +
    "; column += 256) { const float value = __bfloat162float(input[base + column]); sum = __fadd_rn(sum, __fmul_rn(value, value)); }\n" +
    "  partial[threadIdx.x] = sum;\n  __syncthreads();\n" +
    "  for (int stride = 128; stride > 0; stride >>= 1) { if (threadIdx.x < stride) partial[threadIdx.x] = __fadd_rn(partial[threadIdx.x], partial[threadIdx.x + stride]); __syncthreads(); }\n" +
    "  const float inverse = rsqrtf(__fadd_rn(__fdiv_rn(partial[0], " +
    hidden_size.to_string() +
    ".0f), " +
    @luna_cuda_pointwise_aot.canonical_cuda_float_literal(epsilon) +
    "));\n" +
    "  for (int column = threadIdx.x; column < " +
    hidden_size.to_string() +
    "; column += 256) { const float value = __bfloat162float(input[base + column]); const float scale = __bfloat162float(weight[column]); output[base + column] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(value, inverse), scale)); }\n}\n",
  )
  {
    operation: RmsNormBf16,
    numerical_contract: OrderedFloat32ReductionBf16RoundV1,
    function_symbol,
    source,
  }
}
