///|
/// Candidate lanes retain the original ordered component/head folds. A query
/// owner selects from these immutable scores with the original stable tie law.
/// Scratch is borrowed from attention: the ordered program ends index reads
/// before attention reuses the allocation.
fn retained_index_source(
  plan : @precision.IndexedSparseAttentionPrecision,
  physical : @index_physical.RetainedIndexPlan,
  symbol : String,
) -> Bytes raise {
  let (heads, dimension, pool, top_k, tail) = plan.index_geometry()
  let (queries, history) = plan.capacity()
  let (_, capacity) = physical.geometry()
  let width = plan.selected_width()
  let selection = stable_selection_source(
    physical,
    pool,
    "selection_limit",
    "candidate_raw_indices[best*\{pool}+offset]",
  )
  let dot = @arithmetic.scalar_expression(
    @precision.SparseExpression::DotUpdate.expression(),
    [
      "dot", "queries[query_base+component]", "candidate_keys[key_base+component]",
    ],
  )
  let relu = @arithmetic.scalar_expression(
    @precision.SparseExpression::ReluScore.expression(),
    ["dot", "dimension_scale"],
  )
  let weighted = @arithmetic.scalar_expression(
    @precision.SparseExpression::WeightedScore.expression(),
    ["activated", "head_weights[query*\{heads}+head]", "head_scale"],
  )
  let tail_source = if tail {
    (
      $|int first_valid=\{history},visible_count=0,tail_layout_valid=1,saw_valid=0;
      $|for(int key=0;key<\{history};++key) {
      $|  if(key_valid[key]!=0) {if(first_valid==\{history})first_valid=key;if(key<=query_position){saw_valid=1;++visible_count;}}
      $|  else if(saw_valid&&key<=query_position)tail_layout_valid=0;
      $|}
      $|const int tail_count=visible_count%\{pool},tail_start=first_valid+visible_count-tail_count;
      $|for(int offset=0;offset<\{pool-1};++offset) {
      $|  const int raw=tail_start+offset;
      $|  if(tail_layout_valid&&offset<tail_count&&raw>=0&&raw<\{history}&&raw<=query_position&&key_valid[raw]!=0)
      $|    selected_indices[output_base+selection_limit*\{pool}+offset]=raw;
      $|}
    )
  } else {
    ""
  }
  let text =
    $|#include <cuda_bf16.h>
    $|#include <stdint.h>
    $|#include <math.h>
    $|extern "C" __global__ void \{symbol}(const int32_t *counts,const int32_t *query_positions,const __nv_bfloat16 *queries,const __nv_bfloat16 *head_weights,const __nv_bfloat16 *candidate_keys,const int32_t *candidate_raw_indices,const int32_t *candidate_valid,const int32_t *candidate_count,const int32_t *key_valid,int32_t *selected_indices,float *scratch) {
    $|  const int query=blockIdx.x,query_count=counts[3],output_base=query*\{width};
    $|  if(query>=\{queries})return;
    $|  for(int i=threadIdx.x;i<\{width};i+=128)selected_indices[output_base+i]=-1;
    $|  if(counts[2]!=1||query_count<=0||query_count>\{queries}||query>=query_count)return;
    $|  const int query_position=query_positions[query],candidates=*candidate_count;
    $|  if(query_position<0||query_position>=\{history}||candidates<0||candidates>\{capacity})return;
    $|  float *scores=scratch+(unsigned long long)query*\{capacity*2}ULL;
    $|  int *available=(int *)(scores+\{capacity});
    $|  const float dimension_scale=rsqrtf(\{dimension}.0f),head_scale=rsqrtf(\{heads}.0f);
    $|  for(int candidate=threadIdx.x;candidate<candidates;candidate+=128) {
    $|    const int final_raw=candidate_raw_indices[candidate*\{pool}+\{pool-1}];
    $|    const bool valid=candidate_valid[candidate]!=0&&final_raw>=0&&final_raw<=query_position;
    $|    available[candidate]=valid;float score=0.0f;
    $|    if(valid)for(int head=0;head<\{heads};++head) {
    $|      float dot=0.0f;const int query_base=(query*\{heads}+head)*\{dimension},key_base=candidate*\{dimension};
    $|      for(int component=0;component<\{dimension};++component)dot=\{dot};
    $|      const float activated=\{relu},weighted=\{weighted};score=__fadd_rn(score,weighted);
    $|    }
    $|    scores[candidate]=score;
    $|  }
    $|  __syncthreads();
    $|  const int selection_limit=candidates<\{top_k/pool}?candidates:\{top_k/pool};
    $|  \{selection}
    $|  if(threadIdx.x!=0)return;
    $|  \{tail_source}
    $|}
  @utf8.encode(text)
}
