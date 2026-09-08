///|
/// Version marker on the admitted producer entrypoint, not an ambient flag.
/// Old modules retain the four-operand v1 reducer without symbol probing.
pub fn segmented_greedy_head_suffix() -> String {
  "_segmented_greedy_v2"
}

///|
pub fn segmented_greedy_partial_symbol() -> String {
  "lunaflux_greedy_bf16_partial_v2"
}

///|
pub fn segmented_greedy_merge_symbol() -> String {
  "lunaflux_greedy_bf16_merge_v2"
}

///|
pub fn plan_segmented_greedy_cuda(
  vocabulary_size : Int,
  maximum_rows : Int,
) -> @luna_sampling_strategy.GreedySegmentPlan raise SamplingBoundaryError {
  @luna_sampling_strategy.plan_greedy_segments(
    vocabulary_size~,
    maximum_rows~,
    segment_width=4096,
  ) catch {
    _ => raise Invalid(VocabularySize)
  }
}

///|
pub fn embedded_segmented_greedy_cuda_source(
  vocabulary_size : Int,
  maximum_rows : Int,
) -> String raise SamplingBoundaryError {
  let plan = plan_segmented_greedy_cuda(vocabulary_size, maximum_rows)
  let (width, segments) = plan.partition()
  let source =
    $|struct LunaGreedySummaryV2 { float value; int token; int invalid; int padding; };
    $|__device__ __forceinline__ void luna_greedy_combine_v2(float &value, int &token, int &invalid, float other, int other_token, int other_invalid) {
    $|  if (other > value || (other == value && other_token < token)) { value = other; token = other_token; }
    $|  invalid = invalid < other_invalid ? invalid : other_invalid;
    $|}
    $|__device__ __forceinline__ void luna_greedy_reduce_v2(float &value, int &token, int &invalid) {
    $|  __shared__ LunaGreedySummaryV2 warp_summaries[8];
    $|  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    $|  for (int delta=16; delta>0; delta>>=1) {
    $|    float other=__shfl_down_sync(0xffffffffU,value,delta);
    $|    int other_token=__shfl_down_sync(0xffffffffU,token,delta);
    $|    int other_invalid=__shfl_down_sync(0xffffffffU,invalid,delta);
    $|    if (lane+delta<32) luna_greedy_combine_v2(value,token,invalid,other,other_token,other_invalid);
    $|  }
    $|  if (lane==0) warp_summaries[warp]={value,token,invalid,0};
    $|  __syncthreads();
    $|  if (warp==0) {
    $|    value=lane<8?warp_summaries[lane].value:-INFINITY;
    $|    token=lane<8?warp_summaries[lane].token:INT_MAX;
    $|    invalid=lane<8?warp_summaries[lane].invalid:INT_MAX;
    $|    for(int delta=16;delta>0;delta>>=1) {
    $|      float other=__shfl_down_sync(0xffffffffU,value,delta);
    $|      int other_token=__shfl_down_sync(0xffffffffU,token,delta);
    $|      int other_invalid=__shfl_down_sync(0xffffffffU,invalid,delta);
    $|      if(lane+delta<32) luna_greedy_combine_v2(value,token,invalid,other,other_token,other_invalid);
    $|    }
    $|  }
    $|}
    $|extern "C" __global__ void lunaflux_greedy_bf16_partial_v2(const int32_t *counts, const int32_t *row_offsets, const __nv_bfloat16 *logits, LunaGreedySummaryV2 *workspace) {
    $|  const int row=blockIdx.x, segment=blockIdx.y;
    $|  if(row>=counts[2] || row>=\{maximum_rows} || segment>=\{segments}) return;
    $|  const int end=row_offsets[row+1];
    $|  if(end<=0 || end>counts[3]) { if(threadIdx.x==0) workspace[(unsigned long long)row*\{segments}+segment]={-INFINITY,INT_MAX,-2,0}; return; }
    $|  const __nv_bfloat16 *input=logits+(unsigned long long)(end-1)*\{vocabulary_size}ULL;
    $|  float value=-INFINITY; int token=INT_MAX, invalid=INT_MAX;
    $|  const unsigned long long start=(unsigned long long)segment*\{width};
    $|  for(int local=threadIdx.x;local<\{width};local+=256) {
    $|    const unsigned long long index=start+local;
    $|    if(index<\{vocabulary_size}ULL) {
    $|      const float item=__bfloat162float(input[index]);
    $|      if(!isfinite(item)) invalid=invalid<(int)index?invalid:(int)index;
    $|      else luna_greedy_combine_v2(value,token,invalid,item,(int)index,INT_MAX);
    $|    }
    $|  }
    $|  luna_greedy_reduce_v2(value,token,invalid);
    $|  if(threadIdx.x==0) workspace[(unsigned long long)row*\{segments}+segment]={value,token,invalid,0};
    $|}
    $|extern "C" __global__ void lunaflux_greedy_bf16_merge_v2(const int32_t *counts, const LunaGreedySummaryV2 *workspace, int32_t *results) {
    $|  const int row=blockIdx.x;
    $|  if(row>=counts[2] || row>=\{maximum_rows}) return;
    $|  float value=-INFINITY; int token=INT_MAX, invalid=INT_MAX;
    $|  for(int segment=threadIdx.x;segment<\{segments};segment+=256) {
    $|    const LunaGreedySummaryV2 item=workspace[(unsigned long long)row*\{segments}+segment];
    $|    luna_greedy_combine_v2(value,token,invalid,item.value,item.token,item.invalid);
    $|  }
    $|  luna_greedy_reduce_v2(value,token,invalid);
    $|  if(threadIdx.x==0) { results[row*2]=invalid==INT_MAX?token:-1; results[row*2+1]=invalid==INT_MAX?-1:invalid; }
    $|}
    $|
  source
}
