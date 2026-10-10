///|
pub struct WindowCompressedKvCudaSource {
  priv plan : @precision.WindowCompressedKvPrecision
  priv source : Bytes
  priv symbols : ReadOnlyArray[String]
}

///|
pub fn window_compressed_kv_source(
  plan : @precision.WindowCompressedKvPrecision,
  symbol_prefix~ : String,
) -> WindowCompressedKvCudaSource raise {
  guard !symbol_prefix.is_empty() &&
    symbol_prefix.length() <= 112 &&
    symbol_prefix[0] is ('a'..='z' | 'A'..='Z' | '_') &&
    symbol_prefix.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_')) else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let (heads, width, window, rows, positions) = plan.window().geometry()
  let capacity = plan.cache_capacity()
  let slots = plan.selected_width()
  // CUDA's default static shared limit is 48 KiB. This implementation keeps
  // scores and a denominator there; the pure read-set plan has no such cap.
  guard (window.to_int64() + slots.to_int64()) * 4L + 4L <= 49152L else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let learned = plan.read_set()
    is @precision.CompressedRowReadSet::Learned(top_k=_)
  let selected_length = if learned {
    "selected_counts[row]"
  } else {
    "(position + 1) / " + plan.compression_ratio().to_string()
  }
  let offset = match plan.coordinates() {
    CacheRelative => "0"
    OffsetRelative => "offsets[row]"
  }
  let selected_row = if learned {
    "selected[(unsigned long long)row * " +
    Int::max(slots, 1).to_string() +
    "ULL + slot] - " +
    offset
  } else {
    "slot"
  }
  let validate_selection = if learned {
    (
      $|    const int visible = (base + row + 1) / \{plan.compression_ratio()};
      $|    const int limit = selected_counts[row]; if (limit < 0 || limit > \{slots} || limit > visible || \{offset} < 0) { history[1] = 1; return; }
      $|    for (int slot = 0; slot < limit; ++slot) {
      $|      const long long index = (long long)selected[(unsigned long long)row * \{Int::max(slots, 1)}ULL + slot] - \{offset};
      $|      if (index < 0 || index >= visible) { history[1] = 1; return; }
      $|    }
    )
  } else {
    ""
  }
  let scale = @float.canonical_cuda_float_literal(plan.score_scale())
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
        $|extern "C" __global__ void \{symbols[0]}(const int32_t *counts, const int32_t *reset, const int32_t *positions, const int32_t *cache_counts, const int32_t *cache_append, const int32_t *selected_counts, const int32_t *selected, const int32_t *offsets, int32_t *history, int32_t *append) {
        $|  if (threadIdx.x || blockIdx.x) return;
        $|  if (reset[0] == 1) { history[0] = 0; history[1] = 0; }
        $|  const int base = history[0], live = counts[3], retained = cache_counts[3]; append[0] = base; append[1] = 0; append[2] = 1;
        $|  if (reset[0] < 0 || reset[0] > 1 || history[1] || base < 0 || base > \{positions} || live < 0 || live > \{rows} || live > \{positions} - base || (live > 0 && counts[2] != 1) || cache_append[2] || retained < 0 || retained > \{capacity} || (retained && cache_counts[2] != 1)) { history[1] = 1; return; }
        $|  if (retained != (base + live) / \{plan.compression_ratio()}) { history[1] = 1; return; }
        $|  for (int row = 0; row < live; ++row) {
        $|    if (positions[row] != base + row) { history[1] = 1; return; }
        $|\{validate_selection}
        $|  }
        $|  append[1] = live; append[2] = 0;
        $|}
        $|extern "C" __global__ void \{symbols[1]}(const int32_t *append, const int32_t *selected_counts, const int32_t *selected, const int32_t *offsets, const __nv_bfloat16 *queries, const __nv_bfloat16 *current_kv, const __nv_bfloat16 *ring, const __nv_bfloat16 *compressed, const float *sink, __nv_bfloat16 *output) {
        $|  const int row = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
        $|  if (row >= \{rows} || head >= \{heads}) return;
        $|  const unsigned long long query_base = ((unsigned long long)row * \{heads}ULL + head) * \{width}ULL;
        $|  if (append[2] || row >= append[1]) { for (int col = lane; col < \{width}; col += blockDim.x) output[query_base + col] = __float2bfloat16_rn(0.0f); return; }
        $|  const int base = append[0], position = base + row, first = position >= \{window - 1} ? position - \{window - 1} : 0, length = position - first + 1;
        $|  const int compressed_length = \{selected_length};
        $|  __shared__ float scores[\{window + slots}]; __shared__ float denominator;
        $|  if (lane == 0) {
        $|    float maximum = sink[head];
        $|    for (int ordinal = 0; ordinal < length + compressed_length; ++ordinal) {
        $|      const __nv_bfloat16 *key;
        $|      if (ordinal < length) { const int absolute = first + ordinal; key = absolute >= base ? current_kv + (unsigned long long)(absolute - base) * \{width}ULL : ring + (unsigned long long)(absolute % \{window}) * \{width}ULL; }
        $|      else { const int slot = ordinal - length, index = \{selected_row}; key = compressed + (unsigned long long)index * \{width}ULL; }
        $|      float dot = 0.0f; for (int col = 0; col < \{width}; ++col) dot = __fadd_rn(dot, __fmul_rn(__bfloat162float(queries[query_base + col]), __bfloat162float(key[col])));
        $|      scores[ordinal] = __fmul_rn(dot, \{scale}); maximum = fmaxf(maximum, scores[ordinal]);
        $|    }
        $|    float sum = expf(__fsub_rn(sink[head], maximum));
        $|    for (int ordinal = 0; ordinal < length + compressed_length; ++ordinal) { scores[ordinal] = expf(__fsub_rn(scores[ordinal], maximum)); sum = __fadd_rn(sum, scores[ordinal]); }
        $|    denominator = sum;
        $|  }
        $|  __syncthreads();
        $|  for (int col = lane; col < \{width}; col += blockDim.x) {
        $|    float sum = 0.0f;
        $|    for (int ordinal = 0; ordinal < length + compressed_length; ++ordinal) {
        $|      const __nv_bfloat16 *key;
        $|      if (ordinal < length) { const int absolute = first + ordinal; key = absolute >= base ? current_kv + (unsigned long long)(absolute - base) * \{width}ULL : ring + (unsigned long long)(absolute % \{window}) * \{width}ULL; }
        $|      else { const int slot = ordinal - length, index = \{selected_row}; key = compressed + (unsigned long long)index * \{width}ULL; }
        $|      sum = __fadd_rn(sum, __fmul_rn(scores[ordinal], __bfloat162float(key[col])));
        $|    }
        $|    output[query_base + col] = __float2bfloat16_rn(__fdiv_rn(sum, denominator));
        $|  }
        $|}
      ),
    ) +
    b"\n" +
    @window.window_shared_kv_transfer_source(plan.window(), symbol_prefix~)
  { plan, source, symbols: ReadOnlyArray::from_array(symbols), }
}

///|
pub fn WindowCompressedKvCudaSource::plan(
  self : Self,
) -> @precision.WindowCompressedKvPrecision {
  self.plan
}

///|
pub fn WindowCompressedKvCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn WindowCompressedKvCudaSource::symbol(self : Self, stage : Int) -> String {
  self.symbols[stage]
}

///|
pub fn WindowCompressedKvCudaSource::launch(
  self : Self,
  stage : Int,
) -> (Int, Int, Int) {
  let (heads, _, window, rows, _) = self.plan.window().geometry()
  match stage {
    1 => (rows, heads, 128)
    2 => (window, 1, 128)
    _ => (1, 1, 1)
  }
}
