///|
/// Physical-page pools are shared; selected indices remain sequence-logical
/// until the attention consumer maps them. Each component retains the same
/// ordered softmax/dot arithmetic as the contiguous path.
fn paged_pool_source(
  plan : @precision.IndexedSparseAttentionPrecision,
  symbol : String,
) -> Bytes raise {
  let domain = plan.paged_history().unwrap()
  let (_, dimension, pool, _, _) = plan.index_geometry()
  let pools = domain.physical_rows() / pool
  let (rows, _) = plan.capacity()
  let (page_rows, pages, history, sequences) = domain.geometry()
  let logit = @arithmetic.scalar_expression(
    @precision.SparseExpression::PoolLogit.expression(),
    [
      "gates[(uint64_t)raw*\{dimension}+component]",
      "ape[offset*\{dimension}+component]",
    ],
  )
  let denominator = @arithmetic.scalar_expression(
    @precision.SparseExpression::DenominatorUpdate.expression(),
    ["denominator", "logit", "maximum"],
  )
  let probability = @arithmetic.scalar_expression(
    @precision.SparseExpression::Probability.expression(),
    ["logit", "maximum", "denominator"],
  )
  let dot = @arithmetic.scalar_expression(
    @precision.SparseExpression::DotUpdate.expression(),
    ["accumulator", "probability", "keys[(uint64_t)raw*\{dimension}+component]"],
  )
  let publish = @arithmetic.scalar_expression(
    @precision.SparseExpression::Publish.expression(),
    ["accumulator"],
  )
  let quantized = plan.index_scoring() is RowScaledE4M3Matrix
  let includes = if quantized { "#include <cuda_fp8.h>" } else { "" }
  let temporary = if quantized {
    "__shared__ __nv_bfloat16 pooled_row[\{dimension}]; __shared__ float maxima[128];"
  } else {
    ""
  }
  let write = if quantized {
    "pooled_row[component]=\{publish};"
  } else {
    "pooled[(uint64_t)candidate*\{dimension}+component]=\{publish};"
  }
  let quantize = if quantized {
    (
      $| __syncthreads();
      $| float maximum=0.0f;for(int component=threadIdx.x;component<\{dimension};component+=blockDim.x)maximum=fmaxf(maximum,fabsf(__bfloat162float(pooled_row[component])));
      $| maxima[threadIdx.x]=maximum;__syncthreads();
      $| for(int stride=64;stride>0;stride>>=1){if(threadIdx.x<stride)maxima[threadIdx.x]=fmaxf(maxima[threadIdx.x],maxima[threadIdx.x+stride]);__syncthreads();}
      $| const float scale=__fdiv_rn(fmaxf(maxima[0],1.0e-12f),448.0f);
      $| unsigned char *encoded=(unsigned char *)pooled+(uint64_t)candidate*\{dimension+4};
      $| for(int component=threadIdx.x;component<\{dimension};component+=blockDim.x)encoded[component]=__nv_cvt_float_to_fp8(__fdiv_rn(__bfloat162float(pooled_row[component]),scale),__NV_SATFINITE,__NV_E4M3);
      $| if(threadIdx.x==0)*((float *)(encoded+\{dimension}))=scale;
    )
  } else {
    ""
  }
  let source =
    $|#include <cuda_bf16.h>
    $|\{includes}
    $|#include <stdint.h>
    $|#include <math.h>
    $|__device__ __forceinline__ int \{symbol}_written_pool(int row,const int32_t *counts,const int32_t *positions,const int32_t *sequences,const int32_t *offsets,const int32_t *pages,const int32_t *generations) {
    $| if(counts[2]<=0||counts[2]>\{sequences}||counts[3]<=0||counts[3]>\{rows}||row>=counts[3])return -1;
    $| const int seq=sequences[row],position=positions[row];
    $| if(seq<0||seq>=counts[2]||position<0||position>=\{history})return -1;
    $| const int first=offsets[seq],end=offsets[seq+1],at=first+position/\{page_rows};
    $| if(first<0||end<first||end>\{domain.table_capacity()}||at>=end)return -1;
    $| const int page=pages[at];if(page<0||page>=\{pages}||generations[at]<=0)return -1;
    $| return (page*\{page_rows}+position%\{page_rows})/\{pool};
    $|}
    $|extern "C" __global__ void \{symbol}(const int32_t *counts,const __nv_bfloat16 *keys,const __nv_bfloat16 *gates,const int32_t *valid,const __nv_bfloat16 *ape,__nv_bfloat16 *pooled,int32_t *raw_indices,int32_t *pool_valid,int32_t *pool_count,const int32_t *query_counts,const int32_t *positions,const int32_t *row_sequences,const int32_t *page_offsets,const int32_t *page_ids,const int32_t *page_generations) {
    $| // Incremental pool-update plan: one owner per physical write footprint.
    $| const int row=blockIdx.x;if(row>=\{rows})return;
    $| if(row==0&&threadIdx.x==0)*pool_count=\{pools};
    $| const int candidate=\{symbol}_written_pool(row,query_counts,positions,row_sequences,page_offsets,page_ids,page_generations);
    $| if(candidate<0)return;
    $| __shared__ int owner;if(threadIdx.x==0)owner=1;__syncthreads();
    $| for(int earlier=threadIdx.x;earlier<row;earlier+=blockDim.x)if(\{symbol}_written_pool(earlier,query_counts,positions,row_sequences,page_offsets,page_ids,page_generations)==candidate)atomicExch(&owner,0);
    $| __syncthreads();if(!owner)return;
    $| const int first=candidate*\{pool};
    $| \{temporary}
    $| const int generation=valid[first];bool ready=generation>0;
    $| for(int i=1;i<\{pool};++i)ready=ready&&valid[first+i]==generation;
    $| if(threadIdx.x==0)pool_valid[candidate]=ready?generation:0;
    $| for(int i=threadIdx.x;i<\{pool};i+=blockDim.x)raw_indices[first+i]=ready?first+i:-1;
    $| for(int component=threadIdx.x;component<\{dimension};component+=blockDim.x) {
    $|  float accumulator=0.0f;
    $|  if(ready){
    $|   float maximum=-INFINITY;
    $|   for(int offset=0;offset<\{pool};++offset){int raw=first+offset;float logit=\{logit};if(logit>maximum)maximum=logit;}
    $|   float denominator=0.0f;
    $|   for(int offset=0;offset<\{pool};++offset){int raw=first+offset;float logit=\{logit};denominator=\{denominator};}
    $|   for(int offset=0;offset<\{pool};++offset){int raw=first+offset;float logit=\{logit};__nv_bfloat16 probability=\{probability};accumulator=\{dot};}
    $|  }
    $|  \{write}
    $| }
    $| \{quantize}
    $|}
  @utf8.encode(source)
}

///|
fn paged_index_source(
  plan : @precision.IndexedSparseAttentionPrecision,
  symbol : String,
  score_plan : @index_physical.CandidateScorePlan?,
  score_symbol : String,
) -> Bytes raise {
  let domain = plan.paged_history().unwrap()
  let (page_rows, pages, history, sequences) = domain.geometry()
  let (heads, dimension, pool, top_k, tail) = plan.index_geometry()
  let (queries, _) = plan.capacity()
  let capacity = history / pool
  let width = plan.selected_width()
  let selection = stable_selection_source(
    @index_physical.RetainedIndexPlan::new(plan),
    pool,
    "limit",
    "best*\{pool}+offset",
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
  let tail_code = if tail {
    (
      $|for(int raw=candidates*\{pool};raw<=position;++raw){
      $| int at=first_page+raw/\{page_rows};if(at>=end_page)break;
      $| int page=page_ids[at];if(page<0||page>=\{pages})continue;
      $| if(key_valid[page*\{page_rows}+raw%\{page_rows}]==page_generations[at])
      $|  selected_indices[output_base+limit*\{pool}+raw-candidates*\{pool}]=raw;
      $|}
    )
  } else {
    ""
  }
  let ports = "(const int32_t *counts,const int32_t *query_positions,const __nv_bfloat16 *queries,const __nv_bfloat16 *head_weights,const __nv_bfloat16 *candidate_keys,const int32_t *candidate_raw_indices,const int32_t *candidate_valid,const int32_t *candidate_count,const int32_t *key_valid,int32_t *selected_indices,float *scratch,const int32_t *row_sequences,const int32_t *page_offsets,const int32_t *page_ids,const int32_t *page_generations)"
  let geometry =
    $| if(counts[2]<=0||counts[2]>\{sequences}||counts[3]<=0||counts[3]>\{queries}||query>=counts[3])return;
    $| const int sequence=row_sequences[query],position=query_positions[query];
    $| if(sequence<0||sequence>=counts[2]||position<0||position>=\{history})return;
    $| const int first_page=page_offsets[sequence],end_page=page_offsets[sequence+1];
    $| if(first_page<0||end_page<first_page||end_page>\{domain.table_capacity()})return;
    $| const int candidates=(position+1)/\{pool},limit=candidates<\{top_k/pool}?candidates:\{top_k/pool};
    $| float *scores=scratch+(unsigned long long)query*\{capacity*2}ULL;int *available=(int *)(scores+\{capacity});
  let (first, end, stride) = match score_plan {
    Some(tile) =>
      (
        "tile*\{tile.geometry().2}+threadIdx.x",
        "min(candidates,(tile+1)*\{tile.geometry().2})",
        "blockDim.x",
      )
    None => ("threadIdx.x", "candidates", "blockDim.x")
  }
  let scores =
    $| const float dimension_scale=rsqrtf(\{dimension}.0f),head_scale=rsqrtf(\{heads}.0f);
    $| for(int candidate=\{first};candidate<\{end};candidate+=\{stride}) {
    $|  const int raw=candidate*\{pool},at=first_page+raw/\{page_rows};
    $|  int physical=-1;if(at<end_page){int page=page_ids[at];if(page>=0&&page<\{pages})physical=page*\{page_rows/pool}+candidate%\{page_rows/pool};}
    $|  bool valid=physical>=0&&candidate_valid[physical]>0&&candidate_valid[physical]==page_generations[at];
    $|  available[candidate]=valid;float score=0.0f;
    $|  if(valid)for(int head=0;head<\{heads};++head){
    $|   float dot=0.0f;const uint64_t query_base=((uint64_t)query*\{heads}+head)*\{dimension},key_base=(uint64_t)physical*\{dimension};
    $|   for(int component=0;component<\{dimension};++component)dot=\{dot};
    $|   const float activated=\{relu},weighted=\{weighted};score=__fadd_rn(score,weighted);
    $|  }
    $|  scores[candidate]=score;
    $| }
  let producer = if plan.index_scoring() is RowScaledE4M3Matrix {
    matrix_index_source(plan, score_symbol, ports, geometry)
  } else {
    match score_plan {
      Some(tile) =>
        (
          $|extern "C" __global__ void \{score_symbol}\{ports} {
          $| const int live=counts[3];if(live<=0||live>\{queries})return;
          $| const int query=blockIdx.x%live,tile=blockIdx.x/live;
          $| if(tile>=\{tile.tiles_per_query()})return;
          $| \{geometry}
          $| \{scores}
          $|}
        )
      None => ""
    }
  }
  let inline_scores = if score_plan is None { scores } else { "" }
  let source =
    $|#include <cuda_bf16.h>
    $|#include <stdint.h>
    $|#include <math.h>
    $|\{producer}
    $|extern "C" __global__ void \{symbol}\{ports} {
    $| const int query=blockIdx.x,output_base=query*\{width};if(query>=\{queries})return;
    $| for(int i=threadIdx.x;i<\{width};i+=blockDim.x)selected_indices[output_base+i]=-1;
    $| \{geometry}
    $| \{inline_scores}
    $| __syncthreads();
    $| \{selection}
    $| if(threadIdx.x!=0)return;
    $| \{tail_code}
    $|}
  @utf8.encode(source)
}
