///|
/// Shared bounded expert-row permutation. The effect is represented by the
/// grouping plan and stored alongside products, never a token-path allocation.
fn grouping_kernel(
  symbol : String,
  rows : Int,
  selected : Int,
  metadata : Int64,
) -> String {
  let jobs = rows * selected
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
