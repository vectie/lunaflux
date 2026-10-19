///|
/// Numerical expressions, independent-axis schedule and CUDA publication are
/// separate inputs. No model name, tuning record or comparison enters lowering.
pub struct SelectedSparseCudaSource {
  priv schedule : @physical.SelectedSparseSchedule
  priv source : Bytes
  priv reuse : @physical.SelectedQueryReuse?
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
  published_slots? : Int,
  paged_history? : @physical.PagedHistoryDomain,
  accumulation? : @precision.AttentionAccumulationLaw = OrderedFloat32,
) -> SelectedSparseCudaSource raise {
  let extent = published_slots.unwrap_or(selected_width)
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
    extent >= 0 &&
    extent <= selected_width &&
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
  let reuse = if accumulation is TiledFloat32 && extent > 0 {
    @physical.SelectedQueryReuse::new(
      queries=maximum_queries,
      heads=query_heads,
      history=maximum_history,
      query_dimension=qk_dimension,
      value_dimension~,
      rows=16,
      shared_budget=49152L,
    )
  } else {
    None
  }
  let slots = schedule.retained_slots()
  let (pairwise_uniqueness, bitmap_uniqueness) = uniqueness_source(
    schedule, extent,
  )
  let dot = expression(DotUpdate, [
    "dot", "queries[qbase + component]", "keys[kbase + component]",
  ])
  let score = expression(Score, ["dot", "scale"])
  let scalar_scores =
    $|for (int item = worker; item < visible; item += blockDim.x) {
    $|  const unsigned long long kbase = ((unsigned long long)retained[item] * \{kv_heads} + kv_head) * \{qk_dimension}ULL;
    $|  float dot = 0.0f;
    $|  for (int component = 0; component < \{qk_dimension}; ++component) dot = \{dot};
    $|  scores[item] = \{score};
    $|}
  let ordered_scores = score_tile_source(
    schedule, qk_dimension, kv_heads, scalar_scores,
  )
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
  let ordered_output =
    $|for (int component = worker; component < \{value_dimension}; component += blockDim.x) {
    $| float accumulator = 0.0f;
    $| for (int item = 0; item < visible; ++item) {
    $|  const __nv_bfloat16 probability = __float2bfloat16_rn(scores[item]);
    $|  const unsigned long long vbase = ((unsigned long long)retained[item] * \{kv_heads} + kv_head) * \{value_dimension}ULL;
    $|  accumulator = \{pv};
    $| }
    $| output[outbase + component] = \{publish};
    $|}
  let (helper, score_body, output_body) = match accumulation {
    OrderedFloat32 => ("", ordered_scores, ordered_output)
    TiledFloat32 =>
      if reuse is Some(_) {
        // The physical cohort plan amortizes matrix work across live queries.
        // Its incomplete-cohort dispatch retains the ordered row program:
        // a one-live-row MMA changes arithmetic without sharing history reads.
        ("", ordered_scores, ordered_output)
      } else {
        matrix_bodies(qk_dimension, value_dimension, kv_heads, symbol)
      }
  }
  let extra_ports = if paged_history is Some(_) {
    ", const int32_t *row_sequences, const int32_t *page_offsets, const int32_t *page_ids, const int32_t *page_generations"
  } else {
    ""
  }
  let (sequence_guard, live_keys, address, validity, retained_key) = match
    paged_history {
    None => ("counts[2] != 1", "*key_count", "", "key_valid[key] == 0", "key")
    Some(domain) => {
      let (page_rows, pages, context, sequences) = domain.geometry()
      guard context == maximum_history else {
        raise @precision.PrecisionPlanError::IllegalPlacement
      }
      (
        "counts[2] <= 0 || counts[2] > \{sequences}",
        "\{maximum_history}",
        (
          $|const int sequence=row_sequences[query];
          $|if(sequence<0||sequence>=counts[2])return;
          $|const int first_page=page_offsets[sequence],end_page=page_offsets[sequence+1];
          $|if(first_page<0||end_page<first_page||end_page>\{domain.table_capacity()})return;
        ),
        "first_page+key/\{page_rows}>=end_page || page_ids[first_page+key/\{page_rows}]<0 || page_ids[first_page+key/\{page_rows}]>=\{pages} || key_valid[page_ids[first_page+key/\{page_rows}]*\{page_rows}+key%\{page_rows}]!=page_generations[first_page+key/\{page_rows}]",
        "page_ids[first_page+key/\{page_rows}]*\{page_rows}+key%\{page_rows}",
      )
    }
  }
  let row_symbol = symbol + "_single"
  let declaration = if reuse is Some(_) {
    "__device__ __forceinline__ void " + row_symbol
  } else {
    "extern \"C\" __global__ void " + symbol
  }
  let source =
    $|#include <cuda_bf16.h>
    $|#include <stdint.h>
    $|#include <math.h>
    $|\{helper}
    $|// retained-score-v1: independent axes; ordered reductions unchanged.
    $|\{declaration}(const int32_t *counts,
    $|  const int32_t *query_positions, const int32_t *key_count,
    $|  const int32_t *key_valid, const __nv_bfloat16 *queries,
    $|  const __nv_bfloat16 *keys, const __nv_bfloat16 *values,
    $|  const int32_t *selected_indices, __nv_bfloat16 *output, void *scratch\{extra_ports}) {
    $|  const int group = blockIdx.x, worker = threadIdx.x;
    $|  if (group >= \{schedule.workgroups()}) return;
    $|  const int query = group / \{query_heads}, head = group % \{query_heads};
    $|  const int kv_head = head / \{query_heads / kv_heads};
    $|  const unsigned long long outbase = (unsigned long long)group * \{value_dimension}ULL;
    $|  for (int component = worker; component < \{value_dimension}; component += blockDim.x)
    $|    output[outbase + component] = __float2bfloat16_rn(0.0f);
    $|  const int query_count = counts[3], keys_live = \{live_keys};
    $|  if (\{sequence_guard} || query_count <= 0 || query_count > \{maximum_queries} ||
    $|      query >= query_count || keys_live <= 0 || keys_live > \{maximum_history}) return;
    $|  const int position = query_positions[query];
    $|  if (position < 0 || position >= keys_live) return;
    $|  \{address}
    $|  const unsigned long long selected_base = (unsigned long long)query * \{selected_width}ULL;
    $|  const unsigned long long region = (unsigned long long)group * \{slots * 2}ULL;
    $|  int *retained = ((int *)scratch) + region;
    $|  float *scores = ((float *)scratch) + region + \{slots};
    $|  __shared__ int row_valid, visible;
    $|  __shared__ float maximum, denominator;
    $|  if (worker == 0) row_valid = 1;
    $|  __syncthreads();
    $|  // Producer-bound selection prefix: storage stride remains \{selected_width}.
    $|  for (int rank = worker; rank < \{extent}; rank += blockDim.x) {
    $|    const int key = selected_indices[selected_base + rank];
    $|    if (key == -1) continue;
    $|    if (key < 0 || key >= keys_live || key > position || \{validity}) {
    $|      atomicExch(&row_valid, 0); continue;
    $|    }
    $|    \{pairwise_uniqueness}
    $|  }
    $|  __syncthreads();
    $|  if (!row_valid) return;
    $|  \{bitmap_uniqueness}
    $|  if (worker == 0) {
    $|    visible = 0;
    $|    for (int rank = 0; rank < \{extent}; ++rank) {
    $|      const int key = selected_indices[selected_base + rank];
    $|      if (key != -1) retained[visible++] = \{retained_key};
    $|    }
    $|  }
    $|  __syncthreads();
    $|  if (visible == 0) return;
    $|  const float scale = rsqrtf(\{qk_dimension}.0f);
    $|  const unsigned long long qbase = (unsigned long long)group * \{qk_dimension}ULL;
    $|  \{score_body}
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
    $|  \{output_body}
    $|}
  if reuse is Some(plan) {
    let cohort_symbol = symbol + "_cohort"
    let cohort_source = @utf8.decode(
      query_reuse_source(
        query_heads, kv_heads, qk_dimension, value_dimension, selected_width, extent,
        maximum_queries, maximum_history, cohort_symbol, paged_history,
      ),
    )
    let extra_names = if paged_history is Some(_) {
      ",row_sequences,page_offsets,page_ids,page_generations"
    } else {
      ""
    }
    let dispatch =
      $|extern "C" __global__ void \{symbol}(const int32_t *counts,const int32_t *query_positions,const int32_t *key_count,const int32_t *key_valid,const __nv_bfloat16 *queries,const __nv_bfloat16 *keys,const __nv_bfloat16 *values,const int32_t *selected_indices,__nv_bfloat16 *output,void *scratch\{extra_ports}){
      $| if(counts[3]>=\{plan.cohort_rows()})\{cohort_symbol}(counts,query_positions,key_count,key_valid,queries,keys,values,selected_indices,output,scratch\{extra_names});
      $| else \{row_symbol}(counts,query_positions,key_count,key_valid,queries,keys,values,selected_indices,output,scratch\{extra_names});
      $|}
    return {
      schedule,
      reuse,
      source: @utf8.encode(source + "\n" + cohort_source + "\n" + dispatch),
    }
  }
  { schedule, reuse: None, source: @utf8.encode(source + "\n"), }
}

///|
pub fn SelectedSparseCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn SelectedSparseCudaSource::scratch_bytes(self : Self) -> Int64 {
  self.reuse
  .map(plan => Int64::max(plan.scratch_bytes(), self.schedule.scratch_bytes()))
  .unwrap_or(self.schedule.scratch_bytes())
}

///|
pub fn SelectedSparseCudaSource::launch(self : Self) -> (Int, Int) {
  (
    self.reuse
    .map(plan => plan.dispatch_workgroups())
    .unwrap_or(self.schedule.workgroups()),
    self.schedule.workers(),
  )
}
