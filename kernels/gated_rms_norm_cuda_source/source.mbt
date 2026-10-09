///|
fn valid_gated_norm_symbol(symbol : StringView) -> Bool {
  !symbol.is_empty() &&
  symbol.length() <= 128 &&
  symbol[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
pub fn sigmoid_gated_rms_norm_bf16_source(
  head_count~ : Int,
  head_dimension~ : Int,
  maximum_rows~ : Int,
  epsilon~ : Double,
  function_symbol~ : String,
) -> GatedRmsNormCudaSource raise GatedRmsNormCudaSourceError {
  if head_count <= 0 || head_count > 65536 {
    raise InvalidHeadCount
  }
  if head_dimension <= 0 || head_dimension > 65536 {
    raise InvalidHeadDimension
  }
  if maximum_rows <= 0 || maximum_rows > 1048576 {
    raise InvalidMaximumRows
  }
  if epsilon.is_nan() || epsilon.is_inf() || epsilon <= 0.0 || epsilon > 1.0 {
    raise InvalidEpsilon
  }
  if !valid_gated_norm_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let vector_width = head_count.to_int64() * head_dimension.to_int64()
  if vector_width <= 0L ||
    vector_width > 2147483647L ||
    vector_width * maximum_rows.to_int64() > 2147483647L {
    raise OutputSizeOverflow
  }
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <stdint.h>\n#include <math.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const __nv_bfloat16 *recurrent_input, const __nv_bfloat16 *gate_input, const __nv_bfloat16 *weight, __nv_bfloat16 *output) {\n" +
    "  if (blockIdx.x != 0 || threadIdx.x != 0) return;\n" +
    "  const int row_count = counts[3]; if (row_count <= 0 || row_count > " +
    maximum_rows.to_string() +
    " ) return;\n" +
    "  for (int row = 0; row < row_count; ++row) {\n" +
    "    for (int head = 0; head < " +
    head_count.to_string() +
    "; ++head) {\n" +
    "      const unsigned long long base = ((unsigned long long)row * " +
    head_count.to_string() +
    "ULL + (unsigned long long)head) * " +
    head_dimension.to_string() +
    "ULL; float sum_sq = 0.0f;\n" +
    "      for (int column = 0; column < " +
    head_dimension.to_string() +
    "; ++column) { const float value = __bfloat162float(recurrent_input[base + column]); sum_sq = __fadd_rn(sum_sq, __fmul_rn(value, value)); }\n" +
    "      const float mean_sq = __fdiv_rn(sum_sq, " +
    head_dimension.to_string() +
    ".0f);\n" +
    "      const float inverse = rsqrtf(__fadd_rn(mean_sq, " +
    @luna_cuda_pointwise_aot.canonical_cuda_float_literal(epsilon) +
    "));\n" +
    "      for (int column = 0; column < " +
    head_dimension.to_string() +
    "; ++column) { const float value = __bfloat162float(recurrent_input[base + column]); const float normalized = __fmul_rn(value, inverse); const float weighted = __fmul_rn(__bfloat162float(weight[column]), normalized); const float gate = __bfloat162float(gate_input[base + column]); const float sigmoid_gate = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-gate))); output[base + column] = __float2bfloat16_rn(__fmul_rn(weighted, sigmoid_gate)); }\n" +
    "    }\n  }\n}\n",
  )
  {
    operation: PerVectorSigmoidGatedRmsNormBf16,
    numerical_contract: OrderedFloat32RmsNormThenSigmoidGateBf16RoundV1,
    head_count,
    head_dimension,
    maximum_rows,
    epsilon,
    function_symbol,
    source,
  }
}
