///|
/// Numerical expressions, independent-axis schedule and CUDA publication are
/// separate inputs. No model name, tuning record or comparison enters lowering.
pub struct SelectedSparseCudaSource {
  priv schedule : @physical.SelectedSparseSchedule
  priv source : Bytes
}

///|
fn expression(
  op : @precision.SparseExpression,
  args : ArrayView[String],
) -> String raise {
  @arithmetic.scalar_expression(op.expression(), args)
}

///|
pub fn selected_sparse_source(
  query_heads~ : Int,
  kv_heads~ : Int,
  qk_dimension~ : Int,
  value_dimension~ : Int,
  selected_width~ : Int,
  maximum_queries~ : Int,
  maximum_history~ : Int,
  symbol~ : String,
) -> SelectedSparseCudaSource raise {
  guard query_heads > 0 &&
    query_heads <= 1024 &&
    kv_heads > 0 &&
    kv_heads <= query_heads &&
    query_heads % kv_heads == 0 &&
    qk_dimension > 0 &&
    qk_dimension <= 65536 &&
    value_dimension > 0 &&
    value_dimension <= 65536 &&
    selected_width > 0 &&
    selected_width <= 65536 &&
    maximum_queries > 0 &&
    maximum_queries <= maximum_history &&
    maximum_history <= 1048576 &&
    symbol.length() > 0 &&
    symbol.length() <= 128 &&
    symbol[0] is ('a'..='z' | 'A'..='Z' | '_') &&
    symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_')) else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  guard @physical.SelectedSparseSchedule::new(
      queries=maximum_queries,
      heads=query_heads,
      selected_width~,
      history=maximum_history,
      workers=256,
    )
    is Some(schedule) else {
    raise @precision.PrecisionPlanError::ArithmeticOverflow
  }
  guard schedule.workgroups().to_int64() * value_dimension.to_int64() <=
    2147483647L else {
    raise @precision.PrecisionPlanError::ArithmeticOverflow
  }
  let slots = schedule.retained_slots()
  let dot = expression(DotUpdate, [
    "dot", "queries[qbase + component]", "keys[kbase + component]",
  ])
  let score = expression(Score, ["dot", "scale"])
  let denominator = expression(DenominatorUpdate, [
    "denominator", "scores[item]", "maximum",
  ])
  let probability = expression(Probability, [
    "scores[item]", "maximum", "denominator",
  ])
  let pv = expression(DotUpdate, [
    "accumulator", "probability", "values[vbase + component]",
  ])
  let publish = expression(Publish, ["accumulator"])
  let source =
    $|#include <cuda_bf16.h>
    $|#include <stdint.h>
    $|#include <math.h>
    $|// retained-score-v1: independent axes; ordered reductions unchanged.
    $|extern "C" __global__ void \{symbol}(const int32_t *counts,
    $|  const int32_t *query_positions, const int32_t *key_count,
    $|  const int32_t *key_valid, const __nv_bfloat16 *queries,
    $|  const __nv_bfloat16 *keys, const __nv_bfloat16 *values,
    $|  const int32_t *selected_indices, __nv_bfloat16 *output, void *scratch) {
    $|  const int group = blockIdx.x, worker = threadIdx.x;
    $|  if (group >= \{schedule.workgroups()}) return;
    $|  const int query = group / \{query_heads}, head = group % \{query_heads};
    $|  const int kv_head = head / \{query_heads / kv_heads};
    $|  const unsigned long long outbase = (unsigned long long)group * \{value_dimension}ULL;
    $|  for (int component = worker; component < \{value_dimension}; component += blockDim.x)
    $|    output[outbase + component] = __float2bfloat16_rn(0.0f);
    $|  const int query_count = counts[3], keys_live = *key_count;
    $|  if (counts[2] != 1 || query_count <= 0 || query_count > \{maximum_queries} ||
    $|      query >= query_count || keys_live <= 0 || keys_live > \{maximum_history}) return;
    $|  const int position = query_positions[query];
    $|  if (position < 0 || position >= keys_live) return;
    $|  const unsigned long long selected_base = (unsigned long long)query * \{selected_width}ULL;
    $|  const unsigned long long region = (unsigned long long)group * \{slots * 2}ULL;
    $|  int *retained = ((int *)scratch) + region;
    $|  float *scores = ((float *)scratch) + region + \{slots};
    $|  __shared__ int row_valid, visible;
    $|  __shared__ float maximum, denominator;
    $|  if (worker == 0) row_valid = 1;
    $|  __syncthreads();
    $|  for (int rank = worker; rank < \{selected_width}; rank += blockDim.x) {
    $|    const int key = selected_indices[selected_base + rank];
    $|    if (key == -1) continue;
    $|    if (key < 0 || key >= keys_live || key > position || key_valid[key] == 0) {
    $|      atomicExch(&row_valid, 0); continue;
    $|    }
    $|    for (int prior = 0; prior < rank; ++prior)
    $|      if (selected_indices[selected_base + prior] == key) { atomicExch(&row_valid, 0); break; }
    $|  }
    $|  __syncthreads();
    $|  if (!row_valid) return;
    $|  if (worker == 0) {
    $|    visible = 0;
    $|    for (int rank = 0; rank < \{selected_width}; ++rank) {
    $|      const int key = selected_indices[selected_base + rank];
    $|      if (key != -1) retained[visible++] = key;
    $|    }
    $|  }
    $|  __syncthreads();
    $|  if (visible == 0) return;
    $|  const float scale = rsqrtf(\{qk_dimension}.0f);
    $|  const unsigned long long qbase = (unsigned long long)group * \{qk_dimension}ULL;
    $|  for (int item = worker; item < visible; item += blockDim.x) {
    $|    const unsigned long long kbase = ((unsigned long long)retained[item] * \{kv_heads} + kv_head) * \{qk_dimension}ULL;
    $|    float dot = 0.0f;
    $|    for (int component = 0; component < \{qk_dimension}; ++component) dot = \{dot};
    $|    scores[item] = \{score};
    $|  }
    $|  __syncthreads(); // all score producers publish before ordered consumers
    $|  if (worker == 0) {
    $|    maximum = -INFINITY;
    $|    for (int item = 0; item < visible; ++item) if (scores[item] > maximum) maximum = scores[item];
    $|    denominator = 0.0f;
    $|    for (int item = 0; item < visible; ++item) denominator = \{denominator};
    $|  }
    $|  __syncthreads(); // score readers retire before score-to-probability reuse
    $|  for (int item = worker; item < visible; item += blockDim.x)
    $|    scores[item] = __bfloat162float(\{probability});
    $|  __syncthreads();
    $|  for (int component = worker; component < \{value_dimension}; component += blockDim.x) {
    $|    float accumulator = 0.0f;
    $|    for (int item = 0; item < visible; ++item) {
    $|      const __nv_bfloat16 probability = __float2bfloat16_rn(scores[item]);
    $|      const unsigned long long vbase = ((unsigned long long)retained[item] * \{kv_heads} + kv_head) * \{value_dimension}ULL;
    $|      accumulator = \{pv};
    $|    }
    $|    output[outbase + component] = \{publish};
    $|  }
    $|}
  { schedule, source: @utf8.encode(source + "\n"), }
}

///|
pub fn SelectedSparseCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn SelectedSparseCudaSource::scratch_bytes(self : Self) -> Int64 {
  self.schedule.scratch_bytes()
}

///|
pub fn SelectedSparseCudaSource::launch(self : Self) -> (Int, Int) {
  (self.schedule.workgroups(), self.schedule.workers())
}
