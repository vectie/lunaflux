///|
pub struct WindowSharedKvCudaSource {
  priv plan : @precision.WindowSharedKvPrecision
  priv source : Bytes
  priv symbols : ReadOnlyArray[String]
}

///|
pub fn window_shared_kv_source(
  plan : @precision.WindowSharedKvPrecision,
  symbol_prefix~ : String,
) -> WindowSharedKvCudaSource raise {
  guard !symbol_prefix.is_empty() &&
    symbol_prefix.length() <= 112 &&
    symbol_prefix[0] is ('a'..='z' | 'A'..='Z' | '_') &&
    symbol_prefix.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_')) else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let (heads, width, window, rows, positions) = plan.geometry()
  // This correctness lowering stores one score per window position in shared
  // memory. The semantic plan does not inherit that CUDA resource restriction.
  guard window <= 4096 else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let symbols = [
    symbol_prefix + "_reserve",
    symbol_prefix + "_attention",
    symbol_prefix + "_copy",
    symbol_prefix + "_publish",
  ]
  let source = @utf8.encode(
    (
      $|#include <cuda_bf16.h>
      $|#include <stdint.h>
      $|#include <math.h>
      $|extern "C" __global__ void \{symbols[0]}(const int32_t *counts, const int32_t *reset, const int32_t *positions, int32_t *history, int32_t *append) {
      $|  if (threadIdx.x || blockIdx.x) return;
      $|  if (reset[0] == 1) { history[0] = 0; history[1] = 0; }
      $|  const int base = history[0], live = counts[3]; append[0] = base; append[1] = 0; append[2] = 1;
    $|  if (reset[0] < 0 || reset[0] > 1 || history[1] || base < 0 || base > \{positions} || live < 0 || live > \{rows} || live > \{positions} - base || (live > 0 && counts[2] != 1)) { history[1] = 1; return; }
      $|  for (int row = 0; row < live; ++row) if (positions[row] != base + row) { history[1] = 1; return; }
      $|  append[1] = live; append[2] = 0;
      $|}
      $|extern "C" __global__ void \{symbols[1]}(const int32_t *append, const __nv_bfloat16 *queries, const __nv_bfloat16 *current_kv, const __nv_bfloat16 *ring, const float *sink, __nv_bfloat16 *output) {
      $|  const int row = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
      $|  if (append[2] || row >= append[1] || head >= \{heads}) return;
      $|  const int base = append[0], position = base + row, first = position >= \{window - 1} ? position - \{window - 1} : 0, length = position - first + 1;
      $|  const unsigned long long query_base = ((unsigned long long)row * \{heads}ULL + head) * \{width}ULL;
      $|  __shared__ float scores[\{window}]; __shared__ float denominator;
      $|  if (lane == 0) {
      $|    float maximum = sink[head];
      $|    for (int slot = 0; slot < length; ++slot) {
      $|      const int absolute = first + slot;
      $|      const __nv_bfloat16 *key = absolute >= base ? current_kv + (unsigned long long)(absolute - base) * \{width}ULL : ring + (unsigned long long)(absolute % \{window}) * \{width}ULL;
      $|      float dot = 0.0f; for (int component = 0; component < \{width}; ++component) dot = __fadd_rn(dot, __fmul_rn(__bfloat162float(queries[query_base + component]), __bfloat162float(key[component])));
      $|      scores[slot] = __fmul_rn(dot, rsqrtf(\{width}.0f)); maximum = fmaxf(maximum, scores[slot]);
      $|    }
      $|    float sum = expf(__fsub_rn(sink[head], maximum));
      $|    for (int slot = 0; slot < length; ++slot) { scores[slot] = expf(__fsub_rn(scores[slot], maximum)); sum = __fadd_rn(sum, scores[slot]); }
      $|    denominator = sum;
      $|  }
      $|  __syncthreads();
      $|  for (int component = lane; component < \{width}; component += blockDim.x) {
      $|    float sum = 0.0f;
      $|    for (int slot = 0; slot < length; ++slot) {
      $|      const int absolute = first + slot;
      $|      const __nv_bfloat16 *key = absolute >= base ? current_kv + (unsigned long long)(absolute - base) * \{width}ULL : ring + (unsigned long long)(absolute % \{window}) * \{width}ULL;
      $|      sum = __fadd_rn(sum, __fmul_rn(scores[slot], __bfloat162float(key[component])));
      $|    }
      $|    output[query_base + component] = __float2bfloat16_rn(__fdiv_rn(sum, denominator));
      $|  }
      $|}
      $|extern "C" __global__ void \{symbols[2]}(const int32_t *append, const uint16_t *current_kv, uint16_t *ring) {
      $|  const int slot = blockIdx.x; if (append[2] || append[1] == 0 || slot >= \{window}) return;
      $|  const int base = append[0], end = base + append[1];
      $|  const int latest = end - 1 - ((end - 1 - slot + \{window}) % \{window});
      $|  if (latest < base || latest >= end) return;
      $|  for (int component = threadIdx.x; component < \{width}; component += blockDim.x) ring[(unsigned long long)slot * \{width}ULL + component] = current_kv[(unsigned long long)(latest - base) * \{width}ULL + component];
      $|}
      $|extern "C" __global__ void \{symbols[3]}(const int32_t *append, int32_t *history) {
      $|  if (threadIdx.x || blockIdx.x || append[2]) return; history[0] = append[0] + append[1];
      $|}
    ),
  )
  { plan, source, symbols: ReadOnlyArray::from_array(symbols), }
}

///|
pub fn WindowSharedKvCudaSource::plan(
  self : Self,
) -> @precision.WindowSharedKvPrecision {
  self.plan
}

///|
pub fn WindowSharedKvCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn WindowSharedKvCudaSource::symbol(self : Self, stage : Int) -> String {
  self.symbols[stage]
}

///|
pub fn WindowSharedKvCudaSource::launch(
  self : Self,
  stage : Int,
) -> (Int, Int, Int) {
  let (heads, _, window, rows, _) = self.plan.geometry()
  match stage {
    1 => (rows, heads, 128)
    2 => (window, 1, 128)
    _ => (1, 1, 1)
  }
}
