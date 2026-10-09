///|
/// Emits correctness-grade inverse scaled YaRN RoPE for full BF16 attention
/// heads `[row,head,head_dimension]`. Adjacent pairs in the non-RoPE prefix are
/// copied as raw BF16 words; only the final `rope_dimension` components are
/// multiplied by conjugated unit-magnitude frequencies.
pub fn inverse_scaled_yarn_attention_output_bf16_source(
  head_count~ : Int,
  head_dimension~ : Int,
  rope_dimension~ : Int,
  maximum_rows~ : Int,
  maximum_positions~ : Int,
  original_positions~ : Int,
  compressed_theta~ : Double,
  scaling_factor~ : Double,
  beta_fast~ : Double,
  beta_slow~ : Double,
  function_symbol~ : String,
) -> DecoderInverseRopeCudaSource raise DecoderScaledRopeCudaSourceError {
  if head_count <= 0 {
    raise InvalidHeadCount
  }
  if head_dimension <= 0 || head_dimension % 2 != 0 || head_dimension > 2048 {
    raise InvalidHeadDimension
  }
  if rope_dimension <= 0 ||
    rope_dimension % 2 != 0 ||
    rope_dimension > head_dimension {
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
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const int32_t *positions, uint16_t *attention_words) {\n" +
    "  const int32_t live_rows = counts[3]; if (live_rows < 0 || live_rows > " +
    maximum_rows.to_string() +
    ") return; const int32_t row = (int32_t)blockIdx.x; if (row >= live_rows) return; const int32_t head = (int32_t)blockIdx.y; if (head >= " +
    head_count.to_string() +
    ") return; const int32_t pair = (int32_t)threadIdx.x; if (pair >= " +
    (head_dimension / 2).to_string() +
    ") return;\n" +
    "  const int32_t position = positions[row]; if (position < 0 || position >= " +
    maximum_positions.to_string() +
    ") return; const unsigned long long base = ((unsigned long long)row * " +
    head_count.to_string() +
    "ULL + (unsigned long long)head) * " +
    head_dimension.to_string() +
    "ULL + (unsigned long long)(pair * 2);\n" +
    "  if (pair < " +
    ((head_dimension - rope_dimension) / 2).to_string() +
    ") { return; } const int32_t rope_pair = pair - " +
    ((head_dimension - rope_dimension) / 2).to_string() +
    ";\n" +
    "  const double correction_fast = ((double)" +
    rope_dimension.to_string() +
    " * log((double)" +
    original_positions.to_string() +
    " / ((double)" +
    beta_fast.to_string() +
    " * 6.2831853071795864769))) / (2.0 * log((double)" +
    compressed_theta.to_string() +
    ")); const double correction_slow = ((double)" +
    rope_dimension.to_string() +
    " * log((double)" +
    original_positions.to_string() +
    " / ((double)" +
    beta_slow.to_string() +
    " * 6.2831853071795864769))) / (2.0 * log((double)" +
    compressed_theta.to_string() +
    "));\n" +
    "  const float low = (float)fmax(0.0, floor(correction_fast)); const float high = (float)fmin(" +
    (rope_dimension - 1).to_string() +
    ".0, ceil(correction_slow)); const float ramp = fminf(1.0f, fmaxf(0.0f, __fdiv_rn(__fadd_rn((float)rope_pair, -low), __fadd_rn(high, __fadd_rn(-low, high == low ? 0.001f : 0.0f))))); const float smooth = __fadd_rn(1.0f, -ramp);\n" +
    "  const float exponent = __fdiv_rn((float)(rope_pair * 2), " +
    rope_dimension.to_string() +
    ".0f); const float base_frequency = __fdiv_rn(1.0f, powf(" +
    theta +
    ", exponent)); const float frequency = __fadd_rn(__fmul_rn(__fdiv_rn(base_frequency, " +
    factor +
    "), __fadd_rn(1.0f, -smooth)), __fmul_rn(base_frequency, smooth)); const float angle = __fmul_rn((float)position, frequency); const float cosine = cosf(angle); const float sine = sinf(angle);\n" +
    "  __nv_bfloat16 *attention = (__nv_bfloat16 *)attention_words; const float left = __bfloat162float(attention[base]); const float right = __bfloat162float(attention[base + 1ULL]); attention[base] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(left, cosine), __fmul_rn(right, sine))); attention[base + 1ULL] = __float2bfloat16_rn(__fadd_rn(-__fmul_rn(left, sine), __fmul_rn(right, cosine)));\n}\n",
  )
  {
    numerical_contract: AdjacentPairInverseF32YarnBf16RoundV1,
    head_count,
    head_dimension,
    rope_dimension,
    maximum_rows,
    maximum_positions,
    function_symbol,
    source,
  }
}
