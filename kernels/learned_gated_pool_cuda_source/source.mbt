///|
pub struct LearnedGatedPoolCudaSource {
  priv plan : @precision.LearnedGatedPoolPrecision
  priv source : Bytes
  priv symbols : ReadOnlyArray[String]
}

///|
pub fn learned_gated_pool_source(
  plan : @precision.LearnedGatedPoolPrecision,
  symbol_prefix~ : String,
) -> LearnedGatedPoolCudaSource raise {
  guard !symbol_prefix.is_empty() &&
    symbol_prefix.length() <= 112 &&
    symbol_prefix[0] is ('a'..='z' | 'A'..='Z' | '_') &&
    symbol_prefix.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_')) else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let (hidden, width, ratio, overlap, rows, positions) = plan.geometry()
  let coefficient = if overlap { 2 } else { 1 }
  let channels = coefficient * width
  let state_rows = coefficient * ratio
  let current = if overlap { ratio } else { 0 }
  let selected_half = if overlap { width } else { 0 }
  let epsilon = @literal.canonical_cuda_float_literal(plan.norm_epsilon())
  let symbols = [
    symbol_prefix + "_reserve",
    symbol_prefix + "_project",
    symbol_prefix + "_pool",
    symbol_prefix + "_norm",
    symbol_prefix + "_publish",
  ]
  let source = @utf8.encode(
    (
      $|#include <cuda_bf16.h>
      $|#include <stdint.h>
      $|#include <math.h>
      $|extern "C" __global__ void \{symbols[0]}(const int32_t *counts, const int32_t *positions, const int32_t *reset, int32_t *history, int32_t *append, int32_t *output_counts, int32_t *output_positions) {
      $|  if (threadIdx.x || blockIdx.x) return;
      $|  if (reset[0] == 1) { history[0] = 0; history[1] = 0; }
      $|  const int base = history[0], live = counts[3];
      $|  append[0] = base; append[1] = 0; append[2] = 1; append[3] = 0; append[4] = 0; append[5] = reset[0];
      $|  for (int i = 0; i < 5; ++i) output_counts[i] = 0;
      $|  if (reset[0] < 0 || reset[0] > 1 || history[1] || base < 0 || base > \{positions} || live < 0 || live > \{rows} || live > \{positions} - base || (live > 0 && counts[2] != 1)) { history[1] = 1; return; }
      $|  for (int row = 0; row < live; ++row) if (positions[row] != base + row) { history[1] = 1; return; }
      $|  const int first = base / \{ratio}, completed = (base + live) / \{ratio} - first;
      $|  append[1] = live; append[2] = 0; append[3] = completed; append[4] = first;
      $|  output_counts[2] = completed ? 1 : 0; output_counts[3] = completed;
      $|  for (int row = 0; row < completed; ++row) output_positions[row] = (first + row) * \{ratio};
      $|}
      $|extern "C" __global__ void \{symbols[1]}(const int32_t *append, const __nv_bfloat16 *input, const __nv_bfloat16 *wkv, const __nv_bfloat16 *wgate, float *kv, float *scores) {
      $|  const int row = blockIdx.x, col = blockIdx.y * blockDim.x + threadIdx.x;
      $|  if (append[2] || row >= append[1] || col >= \{channels}) return;
      $|  float value = 0.0f, score = 0.0f;
      $|  for (int inner = 0; inner < \{hidden}; ++inner) {
      $|    const float x = __bfloat162float(input[(unsigned long long)row * \{hidden}ULL + inner]);
      $|    const unsigned long long offset = (unsigned long long)col * \{hidden}ULL + inner;
      $|    value = __fadd_rn(value, __fmul_rn(x, __bfloat162float(wkv[offset])));
      $|    score = __fadd_rn(score, __fmul_rn(x, __bfloat162float(wgate[offset])));
      $|  }
      $|  kv[(unsigned long long)row * \{channels}ULL + col] = value; scores[(unsigned long long)row * \{channels}ULL + col] = score;
      $|}
      $|extern "C" __global__ void \{symbols[2]}(const int32_t *append, const float *kv, const float *scores, const float *ape, float *kv_state, float *score_state, __nv_bfloat16 *pooled) {
      $|  const int col = blockIdx.x * blockDim.x + threadIdx.x; if (append[2] || col >= \{width}) return;
      $|  if (append[5]) for (int slot = 0; slot < \{state_rows}; ++slot) for (int half = 0; half < \{coefficient}; ++half) { const unsigned long long offset = (unsigned long long)slot * \{channels}ULL + half * \{width}ULL + col; kv_state[offset] = 0.0f; score_state[offset] = -INFINITY; }
      $|  int published = 0;
      $|  for (int row = 0; row < append[1]; ++row) {
      $|    const int position = append[0] + row, local = position % \{ratio};
      $|    for (int half = 0; half < \{coefficient}; ++half) {
      $|      const int channel = half * \{width} + col; const unsigned long long destination = (unsigned long long)(\{current} + local) * \{channels}ULL + channel, source = (unsigned long long)row * \{channels}ULL + channel;
      $|      kv_state[destination] = kv[source]; score_state[destination] = __fadd_rn(scores[source], ape[(unsigned long long)local * \{channels}ULL + channel]);
      $|    }
      $|    if ((position + 1) % \{ratio}) continue;
      $|    float maximum = -INFINITY;
      $|    for (int slot = 0; slot < \{state_rows}; ++slot) { const int channel = slot < \{current} ? col : \{selected_half} + col; maximum = fmaxf(maximum, score_state[(unsigned long long)slot * \{channels}ULL + channel]); }
      $|    float denominator = 0.0f;
      $|    for (int slot = 0; slot < \{state_rows}; ++slot) { const int channel = slot < \{current} ? col : \{selected_half} + col; denominator = __fadd_rn(denominator, expf(__fsub_rn(score_state[(unsigned long long)slot * \{channels}ULL + channel], maximum))); }
      $|    float value = 0.0f;
      $|    for (int slot = 0; slot < \{state_rows}; ++slot) { const int channel = slot < \{current} ? col : \{selected_half} + col; const unsigned long long offset = (unsigned long long)slot * \{channels}ULL + channel; value = __fadd_rn(value, __fmul_rn(kv_state[offset], __fdiv_rn(expf(__fsub_rn(score_state[offset], maximum)), denominator))); }
      $|    pooled[(unsigned long long)published++ * \{width}ULL + col] = __float2bfloat16_rn(value);
      $|    if (\{if overlap { 1 } else { 0 }}) for (int slot = 0; slot < \{ratio}; ++slot) for (int half = 0; half < \{coefficient}; ++half) { const unsigned long long destination = (unsigned long long)slot * \{channels}ULL + half * \{width}ULL + col, source = destination + \{ratio * channels}ULL; kv_state[destination] = kv_state[source]; score_state[destination] = score_state[source]; }
      $|  }
      $|}
      $|extern "C" __global__ void \{symbols[3]}(const int32_t *append, const __nv_bfloat16 *pooled, const __nv_bfloat16 *norm, __nv_bfloat16 *output) {
      $|  const int row = blockIdx.x; if (append[2] || row >= append[3]) return;
      $|  __shared__ float inverse;
      $|  if (threadIdx.x == 0) { float sum = 0.0f; for (int col = 0; col < \{width}; ++col) { const float x = __bfloat162float(pooled[(unsigned long long)row * \{width}ULL + col]); sum = __fadd_rn(sum, __fmul_rn(x, x)); } inverse = rsqrtf(__fadd_rn(__fdiv_rn(sum, \{width}.0f), \{epsilon})); }
      $|  __syncthreads();
      $|  for (int col = threadIdx.x; col < \{width}; col += blockDim.x) output[(unsigned long long)row * \{width}ULL + col] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(pooled[(unsigned long long)row * \{width}ULL + col]), inverse), __bfloat162float(norm[col])));
      $|}
      $|extern "C" __global__ void \{symbols[4]}(const int32_t *append, int32_t *history) { if (threadIdx.x || blockIdx.x || append[2]) return; history[0] = append[0] + append[1]; }
    ),
  )
  { plan, source, symbols: ReadOnlyArray::from_array(symbols), }
}

///|
pub fn LearnedGatedPoolCudaSource::plan(
  self : Self,
) -> @precision.LearnedGatedPoolPrecision {
  self.plan
}

///|
pub fn LearnedGatedPoolCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn LearnedGatedPoolCudaSource::symbol(self : Self, stage : Int) -> String {
  self.symbols[stage]
}

///|
pub fn LearnedGatedPoolCudaSource::launch(
  self : Self,
  stage : Int,
) -> (Int, Int, Int) {
  let (_, width, _, overlap, rows, _) = self.plan.geometry()
  match stage {
    1 => (rows, (width * (if overlap { 2 } else { 1 }) + 255) / 256, 256)
    2 => ((width + 255) / 256, 1, 256)
    3 => (self.plan.compressed_rows(), 1, 256)
    _ => (1, 1, 1)
  }
}
