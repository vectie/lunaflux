///|
pub struct CompressedIndexSelectionCudaSource {
  priv plan : @precision.CompressedIndexSelectionPrecision
  priv source : Bytes
  priv symbols : ReadOnlyArray[String]
}

///|
pub fn compressed_index_selection_source(
  plan : @precision.CompressedIndexSelectionPrecision,
  symbol_prefix~ : String,
) -> CompressedIndexSelectionCudaSource raise {
  guard !symbol_prefix.is_empty() &&
    symbol_prefix.length() <= 112 &&
    symbol_prefix[0] is ('a'..='z' | 'A'..='Z' | '_') &&
    symbol_prefix.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_')) else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let (heads, width, ratio, rows, positions, _) = plan.geometry()
  let capacity = plan.cache_capacity()
  let physical = Int::max(capacity, 1)
  let slots = Int::max(plan.selected_width(), 1)
  let symbols = [symbol_prefix + "_scores", symbol_prefix + "_select"]
  let offset = match plan.coordinates() {
    CacheRelative => "0"
    OffsetRelative => "offsets[row]"
  }
  let source = @utf8.encode(
    (
      $|#include <cuda_bf16.h>
      $|#include <stdint.h>
      $|#include <limits.h>
      $|#include <math.h>
      $|extern "C" __global__ void \{symbols[0]}(const int32_t *counts, const int32_t *positions, const int32_t *cache_counts, const int32_t *cache_append, const __nv_bfloat16 *queries, const __nv_bfloat16 *head_weights, const __nv_bfloat16 *keys, __nv_bfloat16 *scores) {
      $|  const int row = blockIdx.x, candidate = blockIdx.y * blockDim.x + threadIdx.x;
      $|  if (row >= \{rows} || candidate >= \{physical}) return;
      $|  const unsigned long long output = (unsigned long long)row * \{physical}ULL + candidate;
      $|  scores[output] = __float2bfloat16_rn(-INFINITY);
      $|  const int live = counts[3], retained = cache_counts[3];
      $|  if (live <= 0 || live > \{rows} || counts[2] != 1 || row >= live || cache_append[2] || retained < 0 || retained > \{capacity} || (retained && cache_counts[2] != 1)) return;
      $|  const int position = positions[row];
      $|  if (position < 0 || position >= \{positions} || candidate >= retained || candidate >= (position + 1) / \{ratio}) return;
      $|  float score = 0.0f;
      $|  for (int head = 0; head < \{heads}; ++head) {
      $|    const unsigned long long query_base = ((unsigned long long)row * \{heads}ULL + head) * \{width}ULL, key_base = (unsigned long long)candidate * \{width}ULL;
      $|    float dot = 0.0f;
      $|    for (int col = 0; col < \{width}; ++col) dot = __fadd_rn(dot, __fmul_rn(__bfloat162float(queries[query_base + col]), __bfloat162float(keys[key_base + col])));
      $|    const float projected = __bfloat162float(__float2bfloat16_rn(dot));
      $|    const float product = __bfloat162float(__float2bfloat16_rn(__fmul_rn(fmaxf(projected, 0.0f), __bfloat162float(head_weights[(unsigned long long)row * \{heads}ULL + head]))));
      $|    score = __fadd_rn(score, product);
      $|  }
      $|  scores[output] = __float2bfloat16_rn(score);
      $|}
      $|extern "C" __global__ void \{symbols[1]}(const int32_t *counts, const int32_t *positions, const int32_t *cache_counts, const int32_t *cache_append, const int32_t *offsets, __nv_bfloat16 *scores, int32_t *selected_counts, int32_t *selected) {
      $|  const int row = blockIdx.x, lane = threadIdx.x; if (row >= \{rows}) return;
      $|  const unsigned long long base = (unsigned long long)row * \{slots}ULL;
      $|  for (int slot = lane; slot < \{slots}; slot += blockDim.x) selected[base + slot] = -1;
      $|  if (lane == 0) selected_counts[row] = 0;
      $|  __syncthreads();
      $|  const int live = counts[3], retained = cache_counts[3];
      $|  if (live <= 0 || live > \{rows} || counts[2] != 1 || row >= live || cache_append[2] || retained < 0 || retained > \{capacity} || (retained && cache_counts[2] != 1)) return;
      $|  const int position = positions[row], offset = \{offset};
      $|  if (position < 0 || position >= \{positions} || offset < 0 || (retained > 0 && offset > INT_MAX - (retained - 1))) return;
      $|  const int visible = min(retained, (position + 1) / \{ratio}), limit = min(visible, \{plan.selected_width()});
      $|  __shared__ float best_scores[256]; __shared__ int best_indices[256];
      $|  for (int rank = 0; rank < limit; ++rank) {
      $|    float best = -INFINITY; int index = -1;
      $|    for (int candidate = lane; candidate < visible; candidate += blockDim.x) {
      $|      const float score = __bfloat162float(scores[(unsigned long long)row * \{physical}ULL + candidate]);
      $|      if (isnan(score) || score == -INFINITY) continue;
      $|      if (index < 0 || score > best || (score == best && candidate < index)) { best = score; index = candidate; }
      $|    }
      $|    best_scores[lane] = best; best_indices[lane] = index; __syncthreads();
      $|    for (int stride = 128; stride > 0; stride >>= 1) {
      $|      if (lane < stride) {
      $|        const int other = best_indices[lane + stride]; const float score = best_scores[lane + stride];
      $|        if (other >= 0 && (best_indices[lane] < 0 || score > best_scores[lane] || (score == best_scores[lane] && other < best_indices[lane]))) { best_indices[lane] = other; best_scores[lane] = score; }
      $|      }
      $|      __syncthreads();
      $|    }
      $|    if (best_indices[0] < 0) break;
      $|    if (lane == 0) { selected[base + rank] = offset + best_indices[0]; selected_counts[row] = rank + 1; scores[(unsigned long long)row * \{physical}ULL + best_indices[0]] = __float2bfloat16_rn(-INFINITY); }
      $|    __syncthreads();
      $|  }
      $|}
      $|
    ),
  )
  { plan, source, symbols: ReadOnlyArray::from_array(symbols), }
}

///|
pub fn CompressedIndexSelectionCudaSource::plan(
  self : Self,
) -> @precision.CompressedIndexSelectionPrecision {
  self.plan
}

///|
pub fn CompressedIndexSelectionCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn CompressedIndexSelectionCudaSource::symbol(
  self : Self,
  stage : Int,
) -> String {
  self.symbols[stage]
}

///|
pub fn CompressedIndexSelectionCudaSource::launch(
  self : Self,
  stage : Int,
) -> (Int, Int, Int) {
  let (_, _, _, rows, _, _) = self.plan.geometry()
  (
    rows,
    if stage == 0 {
      (Int::max(self.plan.cache_capacity(), 1) + 255) / 256
    } else {
      1
    },
    256,
  )
}
