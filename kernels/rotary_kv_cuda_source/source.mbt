///|
/// Physical layout is `[row,head,full_width]` for both Q and shared KV.
/// The two functions are borrowed by the surrounding ordered executor.
pub struct RotaryKvCudaSource {
  priv plan : @precision.RotaryKvPrecision
  priv source : Bytes
  priv symbols : ReadOnlyArray[String]
}

///|
fn valid_symbol(symbol : StringView) -> Bool {
  !symbol.is_empty() &&
  symbol.length() <= 112 &&
  symbol[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
pub fn rotary_kv_source(
  plan : @precision.RotaryKvPrecision,
  symbol_prefix~ : String,
) -> RotaryKvCudaSource raise {
  guard valid_symbol(symbol_prefix) else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let (qh, kh, width, rotary, rows, positions) = plan.geometry()
  let prefix = width - rotary
  let symbols = [symbol_prefix + "_rotary", symbol_prefix + "_simulate"]
  let frequency = rotary_frequency_source(plan.frequency(), rotary)
  let product = rotary_product_source(plan.product_law())
  let (block, floor, scale) = plan.simulation()
  let simulation = cache_simulation_source(
    symbols[1],
    rows,
    kh,
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
      $|extern "C" __global__ void \{symbols[0]}(const int32_t *counts, const int32_t *positions, const uint16_t *query, const uint16_t *kv, uint16_t *query_out, uint16_t *kv_out) {
      $|  const int row = (int)blockIdx.x, head = (int)blockIdx.y, live = counts[3];
      $|  if (live <= 0 || live > \{rows} || row >= live || head >= \{qh + kh}) return;
      $|  const int position = positions[row]; if (position < 0 || position >= \{positions}) return;
      $|  const bool is_query = head < \{qh}; const int heads = is_query ? \{qh} : \{kh}; const int local_head = is_query ? head : head - \{qh};
      $|  const unsigned long long base = ((unsigned long long)row * heads + local_head) * \{width}ULL;
      $|  const uint16_t *input_words = is_query ? query : kv; uint16_t *output_words = is_query ? query_out : kv_out;
      $|  for (int col = threadIdx.x; col < \{prefix}; col += blockDim.x) output_words[base + col] = input_words[base + col];
      $|  for (int pair = threadIdx.x; pair < \{rotary / 2}; pair += blockDim.x) {
      $|    \{frequency}
      $|    const float angle = __fmul_rn((float)position, frequency), cosine = cosf(angle), sine = sinf(angle);
      $|    const unsigned long long index = base + \{prefix}ULL + pair * 2ULL;
      $|    const __nv_bfloat16 *input = (const __nv_bfloat16 *)input_words; __nv_bfloat16 *output = (__nv_bfloat16 *)output_words;
      $|    const float left = __bfloat162float(input[index]), right = __bfloat162float(input[index + 1ULL]);
      $|\{product}
      $|  }
      $|}
      $|\{simulation}
      $|
    ),
  )
  { plan, source, symbols: ReadOnlyArray::from_array(symbols), }
}

///|
pub fn RotaryKvCudaSource::plan(self : Self) -> @precision.RotaryKvPrecision {
  self.plan
}

///|
pub fn RotaryKvCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn RotaryKvCudaSource::symbol(self : Self, stage : Int) -> String {
  self.symbols[stage]
}

///|
pub fn RotaryKvCudaSource::launch(self : Self, stage : Int) -> (Int, Int, Int) {
  let (qh, kh, width, rotary, rows, _) = self.plan.geometry()
  let (block, _, _) = self.plan.simulation()
  if stage == 0 {
    (rows, qh + kh, 256)
  } else {
    (rows, Int::max(1, kh * ((width - rotary) / block)), block)
  }
}
