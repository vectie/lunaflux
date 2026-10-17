///|
/// Shared bounded expert-row permutation. The effect is represented by the
/// grouping plan and stored alongside products, never a token-path allocation.
fn grouping_kernel(
  symbol : String,
  rows : Int,
  selected : Int,
  metadata : Int64,
) -> String raise {
  let jobs = rows * selected
  let assignment = @matrix.ExpertGroupingPlan::new(
    maximum_rows=rows,
    selected_experts=selected,
  ).stable_assignment()
  // The private backend bounds local storage; larger capacities retain the
  // existing implementation. All decisions depend only on immutable geometry.
  if assignment.scratch_elements() <= 3072 {
    return parallel_grouping_kernel(
      symbol, rows, selected, metadata, assignment,
    )
  }
  (
    $|extern "C" __global__ void \{symbol}(const int *counts,const int *indices,__nv_bfloat16 *activation) {
    $|  int *meta=(int *)((unsigned char *)activation+\{metadata}ULL);
    $|  for(int i=threadIdx.x;i<\{1+jobs*17};i+=128) meta[i]=-1;
    $|  __syncthreads(); if(threadIdx.x!=0) return;
    $|  meta[0]=0; int rows=counts[3];if(rows<=0||rows>\{rows}) return;
    $|  int live=rows*\{selected},groups=0;
    $|  for(int first=0;first<live;++first) {
    $|    int expert=indices[first];bool seen=false;
    $|    for(int j=0;j<first;++j) if(indices[j]==expert) {seen=true;break;}
    $|    if(seen) continue;int matched=0;
    $|    for(int j=first;j<live;++j) if(indices[j]==expert) {
    $|      int group=groups+matched/16;meta[1+group]=expert;meta[1+\{jobs}+group*16+matched%16]=j;++matched;
    $|    } groups+=(matched+15)/16;
    $|  } meta[0]=groups;
    $|}
  )
}

///|
fn parallel_grouping_kernel(
  symbol : String,
  rows : Int,
  selected : Int,
  metadata : Int64,
  assignment : @matrix.StableAssignmentPlan,
) -> String {
  let jobs = assignment.capacity()
  let tile_rows = assignment.rows_per_group()
  (
    $|extern "C" __global__ void \{symbol}(const int *counts,const int *indices,__nv_bfloat16 *activation) {
    $|  int *meta=(int *)((unsigned char *)activation+\{metadata}ULL);
    $|  __shared__ int first[\{jobs}],ordinal[\{jobs}],offset[\{jobs}];
    $|  for(int i=threadIdx.x;i<\{1+jobs*17};i+=128) meta[i]=-1;
    $|  int rows=counts[3],live=rows>0&&rows<=\{rows}?rows*\{selected}:0;
    $|  for(int job=threadIdx.x;job<live;job+=128) {
    $|    int expert=indices[job],leader=job,rank=0,total=0;
    $|    for(int other=0;other<live;++other) if(indices[other]==expert) {
    $|      if(other<leader) leader=other;if(other<job) ++rank;++total;
    $|    }
    $|    first[job]=leader;ordinal[job]=rank;
    $|    offset[job]=leader==job?(total+\{tile_rows-1})/\{tile_rows}:0;
    $|  }
    $|  __syncthreads();
    $|  if(threadIdx.x==0) {
    $|    int groups=0;for(int job=0;job<live;++job) {int count=offset[job];offset[job]=groups;groups+=count;}meta[0]=groups;
    $|  }
    $|  __syncthreads();
    $|  for(int job=threadIdx.x;job<live;job+=128) {
    $|    int rank=ordinal[job],group=offset[first[job]]+rank/\{tile_rows};
    $|    if(rank%\{tile_rows}==0) meta[1+group]=indices[job];
    $|    meta[1+\{jobs}+group*\{tile_rows}+rank%\{tile_rows}]=job;
    $|  }
    $|}
  )
}
