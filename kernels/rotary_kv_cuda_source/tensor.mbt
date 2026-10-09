///|
/// Shared standalone suffix-rotation lowering. Prefix words are preserved,
/// and BF16 rounding precedes any later consumer's transform or simulation.
pub fn rotary_tensor_source(
  plan : @precision.RotaryTensorPrecision,
  symbol~ : String,
) -> String raise {
  guard valid_symbol(symbol) else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let (heads, width, rotary, rows, positions) = plan.geometry()
  let frequency = rotary_frequency_source(plan.frequency(), rotary)
  let sign = if plan.inverse() { "-" } else { "" }
  (
    $|extern "C" __global__ void \{symbol}(const int32_t *counts, const int32_t *positions, const uint16_t *input_words, uint16_t *output_words) {
    $|  const int row = (int)blockIdx.x, head = (int)blockIdx.y, live = counts[3];
    $|  if (live <= 0 || live > \{rows} || row >= live || head >= \{heads}) return;
    $|  const int position = positions[row]; if (position < 0 || position >= \{positions}) return;
    $|  const unsigned long long base = ((unsigned long long)row * \{heads}ULL + head) * \{width}ULL;
    $|  for (int col = threadIdx.x; col < \{width - rotary}; col += blockDim.x) output_words[base + col] = input_words[base + col];
    $|  for (int pair = threadIdx.x; pair < \{rotary / 2}; pair += blockDim.x) {
    $|    \{frequency}
    $|    const float angle = __fmul_rn((float)position, frequency), cosine = cosf(angle), sine = \{sign}sinf(angle);
    $|    const unsigned long long index = base + \{width - rotary}ULL + pair * 2ULL;
    $|    const __nv_bfloat16 *input = (const __nv_bfloat16 *)input_words; __nv_bfloat16 *output = (__nv_bfloat16 *)output_words;
    $|    const float left = __bfloat162float(input[index]), right = __bfloat162float(input[index + 1ULL]);
    $|    output[index] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(left, cosine), -__fmul_rn(right, sine)));
    $|    output[index + 1ULL] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(left, sine), __fmul_rn(right, cosine)));
    $|  }
    $|}
  )
}
