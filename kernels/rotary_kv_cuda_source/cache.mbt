///|
/// Standalone compact rows, without redundant query rotation/allocation.
pub struct RotaryCacheCudaSource {
  priv plan : @precision.RotaryCachePrecision
  priv source : Bytes
  priv symbols : ReadOnlyArray[String]
}

///|
pub fn rotary_cache_source(
  plan : @precision.RotaryCachePrecision,
  symbol_prefix~ : String,
) -> RotaryCacheCudaSource raise {
  guard valid_symbol(symbol_prefix) else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let (heads, width, rotary, rows, positions) = plan.geometry()
  let prefix = width - rotary
  let symbols = [symbol_prefix + "_rotary", symbol_prefix + "_simulate"]
  let frequency = rotary_frequency_source(plan.frequency(), rotary)
  let (block, floor, scale) = plan.simulation()
  let simulation = cache_simulation_source(
    symbols[1],
    rows,
    heads,
    width,
    prefix,
    block,
    floor,
    scale,
  )
  let source = @utf8.encode(
    (
      $|#include <cuda.h>
      $|#include <cuda_bf16.h>
      $|#include <cuda_fp8.h>
      $|#include <math.h>
      $|#include <stdint.h>
      $|extern "C" __global__ void \{symbols[0]}(const int32_t *counts, const int32_t *positions, const uint16_t *input_words, uint16_t *output_words) {
      $|  const int row = (int)blockIdx.x, head = (int)blockIdx.y, live = counts[3];
      $|  if (live <= 0 || live > \{rows} || row >= live || head >= \{heads}) return;
      $|  const int position = positions[row]; if (position < 0 || position >= \{positions}) return;
      $|  const unsigned long long base = ((unsigned long long)row * \{heads}ULL + head) * \{width}ULL;
      $|  for (int col = threadIdx.x; col < \{prefix}; col += blockDim.x) output_words[base + col] = input_words[base + col];
      $|  for (int pair = threadIdx.x; pair < \{rotary / 2}; pair += blockDim.x) {
      $|    \{frequency}
      $|    const float angle = __fmul_rn((float)position, frequency), cosine = cosf(angle), sine = sinf(angle);
      $|    const unsigned long long index = base + \{prefix}ULL + pair * 2ULL;
      $|    const __nv_bfloat16 *input = (const __nv_bfloat16 *)input_words; __nv_bfloat16 *output = (__nv_bfloat16 *)output_words;
      $|    const float left = __bfloat162float(input[index]), right = __bfloat162float(input[index + 1ULL]);
      $|    output[index] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(left, cosine), -__fmul_rn(right, sine)));
      $|    output[index + 1ULL] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(left, sine), __fmul_rn(right, cosine)));
      $|  }
      $|}
      $|\{simulation}
    ),
  )
  { plan, source, symbols: ReadOnlyArray::from_array(symbols), }
}

///|
pub fn RotaryCacheCudaSource::plan(
  self : Self,
) -> @precision.RotaryCachePrecision {
  self.plan
}

///|
pub fn RotaryCacheCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn RotaryCacheCudaSource::symbol(self : Self, stage : Int) -> String {
  self.symbols[stage]
}

///|
pub fn RotaryCacheCudaSource::launch(
  self : Self,
  stage : Int,
) -> (Int, Int, Int) {
  let (heads, width, rotary, rows, _) = self.plan.geometry()
  let (block, _, _) = self.plan.simulation()
  if stage == 0 {
    (rows, heads, 256)
  } else {
    (rows, Int::max(1, heads * ((width - rotary) / block)), block)
  }
}
