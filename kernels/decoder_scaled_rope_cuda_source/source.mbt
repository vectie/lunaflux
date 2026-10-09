///|
fn valid_scaled_rope_symbol(symbol : StringView) -> Bool {
  if symbol.is_empty() || symbol.length() > 128 {
    return false
  }
  symbol[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
fn finite_positive(value : Double) -> Bool {
  !value.is_nan() && !value.is_inf() && value > 0.0
}

///|
/// Emit correctness-grade scaled YaRN RoPE over adjacent BF16 pairs. Query and
/// key/value operands contain only their RoPE suffixes, laid out as
/// `[row, head, rope_dimension]`; unrelated head components are excluded.
pub fn scaled_yarn_rope_bf16_source(
  query_head_count~ : Int,
  key_value_head_count~ : Int,
  rope_dimension~ : Int,
  maximum_rows~ : Int,
  maximum_positions~ : Int,
  original_positions~ : Int,
  compressed_theta~ : Double,
  scaling_factor~ : Double,
  beta_fast~ : Double,
  beta_slow~ : Double,
  function_symbol~ : String,
) -> DecoderScaledRopeCudaSource raise DecoderScaledRopeCudaSourceError {
  if query_head_count <= 0 || key_value_head_count <= 0 {
    raise InvalidHeadCount
  }
  if rope_dimension <= 0 || rope_dimension % 2 != 0 || rope_dimension > 1024 {
    raise InvalidRopeDimension
  }
  if maximum_rows <= 0 || maximum_rows > 1048576 {
    raise InvalidMaximumRows
  }
  if maximum_positions <= 0 {
    raise InvalidMaximumPositions
  }
  if original_positions <= 0 || original_positions > maximum_positions {
    raise InvalidOriginalPositions
  }
  if !finite_positive(compressed_theta) {
    raise InvalidTheta
  }
  if !finite_positive(scaling_factor) {
    raise InvalidScaling
  }
  if !finite_positive(beta_fast) || !finite_positive(beta_slow) {
    raise InvalidBeta
  }
  if !valid_scaled_rope_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let theta = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(
    compressed_theta,
  )
  let factor = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(
    scaling_factor,
  )
  let theta_double = compressed_theta.to_string()
  let fast_double = beta_fast.to_string()
  let slow_double = beta_slow.to_string()
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const int32_t *positions, const __nv_bfloat16 *query_input, __nv_bfloat16 *query_output, const __nv_bfloat16 *key_value_input, __nv_bfloat16 *key_value_output) {\n" +
    "  const int32_t live_rows = counts[3];\n  if (live_rows < 0 || live_rows > " +
    maximum_rows.to_string() +
    ") return;\n" +
    "  const int32_t row = (int32_t)blockIdx.x;\n  if (row >= live_rows) return;\n" +
    "  const int32_t position = positions[row];\n  if (position < 0 || position >= " +
    maximum_positions.to_string() +
    ") return;\n" +
    "  const int32_t head = (int32_t)blockIdx.y;\n  const int32_t pair = (int32_t)threadIdx.x;\n  if (pair >= " +
    (rope_dimension / 2).to_string() +
    ") return;\n" +
    "  const double correction_fast = ((double)" +
    rope_dimension.to_string() +
    " * log((double)" +
    original_positions.to_string() +
    " / ((double)" +
    fast_double +
    " * 6.2831853071795864769))) / (2.0 * log((double)" +
    theta_double +
    "));\n" +
    "  const double correction_slow = ((double)" +
    rope_dimension.to_string() +
    " * log((double)" +
    original_positions.to_string() +
    " / ((double)" +
    slow_double +
    " * 6.2831853071795864769))) / (2.0 * log((double)" +
    theta_double +
    "));\n" +
    "  const float low = (float)fmax(0.0, floor(correction_fast));\n  const float high = (float)fmin(" +
    (rope_dimension - 1).to_string() +
    ".0, ceil(correction_slow));\n" +
    "  const float ramp = fminf(1.0f, fmaxf(0.0f, __fdiv_rn(__fadd_rn((float)pair, -low), __fadd_rn(high, __fadd_rn(-low, high == low ? 0.001f : 0.0f)))));\n" +
    "  const float smooth = __fadd_rn(1.0f, -ramp);\n" +
    "  const float exponent = __fdiv_rn((float)(pair * 2), " +
    rope_dimension.to_string() +
    ".0f);\n" +
    "  const float base_frequency = __fdiv_rn(1.0f, powf(" +
    theta +
    ", exponent));\n" +
    "  const float frequency = __fadd_rn(__fmul_rn(__fdiv_rn(base_frequency, " +
    factor +
    "), __fadd_rn(1.0f, -smooth)), __fmul_rn(base_frequency, smooth));\n" +
    "  const float angle = __fmul_rn((float)position, frequency);\n  const float cosine = cosf(angle);\n  const float sine = sinf(angle);\n" +
    "  const int32_t query_heads = " +
    query_head_count.to_string() +
    ";\n  const int32_t kv_heads = " +
    key_value_head_count.to_string() +
    ";\n" +
    "  const bool query = head < query_heads;\n  if (!query && head >= query_heads + kv_heads) return;\n" +
    "  const int32_t local_head = query ? head : head - query_heads;\n  const unsigned long long base = ((unsigned long long)row * (unsigned long long)(query ? query_heads : kv_heads) + (unsigned long long)local_head) * " +
    rope_dimension.to_string() +
    "ULL + (unsigned long long)(pair * 2);\n" +
    "  const __nv_bfloat16 *input = query ? query_input : key_value_input;\n  __nv_bfloat16 *output = query ? query_output : key_value_output;\n" +
    "  const float left = __bfloat162float(input[base]);\n  const float right = __bfloat162float(input[base + 1ULL]);\n" +
    "  output[base] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(left, cosine), -__fmul_rn(right, sine)));\n" +
    "  output[base + 1ULL] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(left, sine), __fmul_rn(right, cosine)));\n}\n",
  )
  {
    numerical_contract: AdjacentPairF32YarnBf16RoundV1,
    query_head_count,
    key_value_head_count,
    rope_dimension,
    maximum_rows,
    maximum_positions,
    function_symbol,
    source,
  }
}
