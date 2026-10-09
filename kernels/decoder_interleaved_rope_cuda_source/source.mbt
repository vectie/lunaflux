///|
fn valid_interleaved_rope_symbol(symbol : StringView) -> Bool {
  !symbol.is_empty() &&
  symbol.length() <= 128 &&
  symbol[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
pub fn interleaved_pair_rope_with_query_prefix_copy_bf16_source(
  query_head_count~ : Int,
  key_head_count~ : Int,
  query_head_dimension~ : Int,
  rope_dimension~ : Int,
  maximum_rows~ : Int,
  maximum_positions~ : Int,
  theta~ : Double,
  function_symbol~ : String,
) -> DecoderInterleavedRopeCudaSource raise DecoderInterleavedRopeCudaSourceError {
  if query_head_count <= 0 || key_head_count <= 0 {
    raise InvalidHeadCount
  }
  if query_head_dimension <= 0 || query_head_dimension > 65536 {
    raise InvalidHeadDimension
  }
  if rope_dimension <= 0 ||
    rope_dimension % 2 != 0 ||
    rope_dimension > 1024 ||
    rope_dimension > query_head_dimension {
    raise InvalidRopeDimension
  }
  if maximum_rows <= 0 || maximum_rows > 1048576 {
    raise InvalidMaximumRows
  }
  if maximum_positions <= 0 {
    raise InvalidMaximumPositions
  }
  if theta.is_nan() || theta.is_inf() || theta <= 0.0 {
    raise InvalidTheta
  }
  if !valid_interleaved_rope_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let prefix_dimension = query_head_dimension - rope_dimension
  let half_rope = rope_dimension / 2
  let theta_literal = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(
    theta,
  )
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const int32_t *positions, const __nv_bfloat16 *query_input, const __nv_bfloat16 *key_input, __nv_bfloat16 *query_output, __nv_bfloat16 *key_output) {\n" +
    "  const int32_t live_rows = counts[3]; const int32_t row = (int32_t)blockIdx.x;\n" +
    "  if (live_rows <= 0 || live_rows > " +
    maximum_rows.to_string() +
    " || row >= live_rows) return;\n" +
    "  const int32_t position = positions[row]; if (position < 0 || position >= " +
    maximum_positions.to_string() +
    " ) return;\n" +
    "  const int32_t head = (int32_t)blockIdx.y; const bool query = head < " +
    query_head_count.to_string() +
    ";\n" +
    "  if (!query && head >= " +
    (query_head_count + key_head_count).to_string() +
    " ) return;\n" +
    "  const int32_t local_head = query ? head : head - " +
    query_head_count.to_string() +
    ";\n" +
    "  const unsigned long long query_base = ((unsigned long long)row * " +
    query_head_count.to_string() +
    "ULL + (unsigned long long)local_head) * " +
    query_head_dimension.to_string() +
    "ULL;\n" +
    "  const unsigned long long key_base = ((unsigned long long)row * " +
    key_head_count.to_string() +
    "ULL + (unsigned long long)local_head) * " +
    rope_dimension.to_string() +
    "ULL;\n" +
    "  if (query) for (int32_t column = (int32_t)threadIdx.x; column < " +
    prefix_dimension.to_string() +
    "; column += 256) query_output[query_base + column] = query_input[query_base + column];\n" +
    "  const int32_t pair = (int32_t)threadIdx.x; if (pair >= " +
    half_rope.to_string() +
    ") return;\n" +
    "  const float exponent = __fdiv_rn((float)(pair * 2), " +
    rope_dimension.to_string() +
    ".0f);\n" +
    "  const float frequency = __fdiv_rn(1.0f, powf(" +
    theta_literal +
    ", exponent)); const float angle = __fmul_rn((float)position, frequency);\n" +
    "  const __nv_bfloat16 cosine_bf16 = __float2bfloat16_rn(cosf(angle)); const __nv_bfloat16 sine_bf16 = __float2bfloat16_rn(sinf(angle));\n" +
    "  const float cosine = __bfloat162float(cosine_bf16); const float sine = __bfloat162float(sine_bf16);\n" +
    "  const unsigned long long input_base = query ? query_base + " +
    prefix_dimension.to_string() +
    "ULL : key_base;\n" +
    "  const unsigned long long output_base = query ? query_base + " +
    prefix_dimension.to_string() +
    "ULL : key_base;\n" +
    "  const __nv_bfloat16 *input = query ? query_input : key_input; __nv_bfloat16 *output = query ? query_output : key_output;\n" +
    "  const float left = __bfloat162float(input[input_base + (unsigned long long)(pair * 2)]);\n" +
    "  const float right = __bfloat162float(input[input_base + (unsigned long long)(pair * 2 + 1)]);\n" +
    "  const __nv_bfloat16 left_cos_bf16 = __float2bfloat16_rn(__fmul_rn(left, cosine)); const __nv_bfloat16 right_sin_bf16 = __float2bfloat16_rn(__fmul_rn(right, sine));\n" +
    "  const __nv_bfloat16 right_cos_bf16 = __float2bfloat16_rn(__fmul_rn(right, cosine)); const __nv_bfloat16 left_sin_bf16 = __float2bfloat16_rn(__fmul_rn(left, sine));\n" +
    "  output[output_base + (unsigned long long)pair] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(left_cos_bf16), -__bfloat162float(right_sin_bf16)));\n" +
    "  output[output_base + " +
    half_rope.to_string() +
    "ULL + (unsigned long long)pair] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(right_cos_bf16), __bfloat162float(left_sin_bf16)));\n}\n",
  )
  {
    operation: InterleavedPairRopeWithQueryPrefixCopyBf16,
    numerical_contract: Float32DefaultFrequencyTrigRoundedBf16PairArithmeticV1,
    query_head_count,
    key_head_count,
    query_head_dimension,
    rope_dimension,
    maximum_rows,
    maximum_positions,
    theta,
    function_symbol,
    source,
  }
}
