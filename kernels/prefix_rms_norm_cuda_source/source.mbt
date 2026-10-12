///|
fn valid_prefix_norm_symbol(symbol : StringView) -> Bool {
  !symbol.is_empty() &&
  symbol.length() <= 128 &&
  symbol[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
pub fn prefix_rms_norm_bf16_with_optional_suffix_copy_source(
  prefix_width~ : Int,
  suffix_width~ : Int,
  maximum_rows~ : Int,
  epsilon~ : Double,
  function_symbol~ : String,
  square_law? : @precision.SquareReductionLaw = VectorizedTreeFloat32(
    partitions=256,
    vector_width=1,
  ),
) -> PrefixRmsNormCudaSource raise PrefixRmsNormCudaSourceError {
  if prefix_width <= 0 || prefix_width > 65536 {
    raise InvalidPrefixWidth
  }
  if suffix_width < 0 || suffix_width > 65536 {
    raise InvalidSuffixWidth
  }
  if maximum_rows <= 0 || maximum_rows > 1048576 {
    raise InvalidMaximumRows
  }
  if epsilon.is_nan() || epsilon.is_inf() || epsilon <= 0.0 || epsilon > 1.0 {
    raise InvalidEpsilon
  }
  if !valid_prefix_norm_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  if square_law !=
    @precision.SquareReductionLaw::VectorizedTreeFloat32(
      partitions=256,
      vector_width=1,
    ) {
    return vector_prefix_norm_source(
      prefix_width~,
      suffix_width~,
      maximum_rows~,
      epsilon~,
      function_symbol~,
      square_law~,
    )
  }
  let suffix_parameter = if suffix_width > 0 {
    ", __nv_bfloat16 *suffix_output"
  } else {
    ""
  }
  let suffix_copy = if suffix_width > 0 {
    "  for (int column = threadIdx.x; column < " +
    suffix_width.to_string() +
    "; column += 256) suffix_output[(unsigned long long)row * " +
    suffix_width.to_string() +
    "ULL + column] = input[input_base + " +
    prefix_width.to_string() +
    "ULL + column];\n"
  } else {
    ""
  }
  let square = rms_expression(SquareUpdate, ["sum", "value"])
  let sum = rms_expression(SumUpdate, [
    "partial[threadIdx.x]", "partial[threadIdx.x + stride]",
  ])
  let inverse = rms_inverse(
    "partial[0]",
    prefix_width,
    @luna_cuda_pointwise_aot.canonical_cuda_float_literal(epsilon),
  )
  let publish = rms_expression(WeightedPublish, ["value", "inverse", "scale"])
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <stdint.h>\n#include <math.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const __nv_bfloat16 *input, const __nv_bfloat16 *weight, __nv_bfloat16 *normalized_output" +
    suffix_parameter +
    ") {\n" +
    "  const int row = (int)blockIdx.x; const int row_count = counts[3];\n" +
    "  if (row_count <= 0 || row_count > " +
    maximum_rows.to_string() +
    " || row >= row_count) return;\n" +
    "  __shared__ float partial[256]; float sum = 0.0f;\n" +
    "  const unsigned long long input_base = (unsigned long long)row * " +
    (prefix_width + suffix_width).to_string() +
    "ULL;\n" +
    "  for (int column = threadIdx.x; column < " +
    prefix_width.to_string() +
    "; column += 256) { const float value = __bfloat162float(input[input_base + column]); sum = \{square}; }\n" +
    "  partial[threadIdx.x] = sum; __syncthreads();\n" +
    "  for (int stride = 128; stride > 0; stride >>= 1) { if (threadIdx.x < stride) partial[threadIdx.x] = \{sum}; __syncthreads(); }\n" +
    "  const float inverse = \{inverse};\n" +
    "  const unsigned long long output_base = (unsigned long long)row * " +
    prefix_width.to_string() +
    "ULL;\n" +
    "  for (int column = threadIdx.x; column < " +
    prefix_width.to_string() +
    "; column += 256) { const float value = __bfloat162float(input[input_base + column]); const float scale = __bfloat162float(weight[column]); normalized_output[output_base + column] = \{publish}; }\n" +
    suffix_copy +
    "}\n",
  )
  {
    operation: PrefixRmsNormBf16WithOptionalSuffixCopy,
    numerical_contract: OrderedFloat32PrefixReductionBf16RoundAndBitExactSuffixCopyV1,
    prefix_width,
    suffix_width,
    maximum_rows,
    epsilon,
    function_symbol,
    source,
    threads: 256,
  }
}
