///|
/// Lower a stable-selection ownership plan. The caller supplies only the
/// publication address (contiguous raw IDs or paged logical IDs). Scores and
/// validity are already published and retain their original ordered arithmetic.
fn stable_selection_source(
  plan : @index_physical.RetainedIndexPlan,
  pool : Int,
  limit : String,
  value : String,
) -> String {
  match plan.selection(128).unwrap() {
    IndependentRanks =>
      (
        $|// stable-topk: independent immutable ranks; ties use candidate ID.
        $|for(int best=threadIdx.x;best<candidates;best+=blockDim.x) {
        $| if(!available[best]||isnan(scores[best]))continue;
        $| const float score=scores[best];int rank=0;
        $| for(int other=0;other<candidates;++other) {
        $|  if(!available[other])continue;
        $|  const float before=scores[other];
        $|  rank+=(before>score||(before==score&&other<best));
        $| }
        $| if(rank<\{limit})for(int offset=0;offset<\{pool};++offset)
        $|  selected_indices[output_base+rank*\{pool}+offset]=\{value};
        $|}
      )
    CooperativeMaxima =>
      (
        $|// stable-topk: bounded-work cooperative maxima for large domains.
        $|__shared__ float winner_scores[128];__shared__ int winner_ids[128];
        $|for(int rank=0;rank<\{limit};++rank) {
        $| int best=-1;float score=-INFINITY;
        $| for(int candidate=threadIdx.x;candidate<candidates;candidate+=blockDim.x) {
        $|  if(!available[candidate])continue;const float x=scores[candidate];
        $|  if(x>score||(x==score&&(best<0||candidate<best))){score=x;best=candidate;}
        $| }
        $| winner_scores[threadIdx.x]=score;winner_ids[threadIdx.x]=best;__syncthreads();
        $| for(int stride=64;stride>0;stride>>=1) {
        $|  if(threadIdx.x<stride) {
        $|   int a=winner_ids[threadIdx.x],b=winner_ids[threadIdx.x+stride];
        $|   float x=winner_scores[threadIdx.x],y=winner_scores[threadIdx.x+stride];
        $|   if(b>=0&&(a<0||y>x||(y==x&&b<a))){winner_ids[threadIdx.x]=b;winner_scores[threadIdx.x]=y;}
        $|  }__syncthreads();
        $| }
        $| best=winner_ids[0];if(best<0)break;
        $| if(threadIdx.x==0)available[best]=0;
        $| for(int offset=threadIdx.x;offset<\{pool};offset+=blockDim.x)
        $|  selected_indices[output_base+rank*\{pool}+offset]=\{value};
        $| __syncthreads(); // publication retires before the next maximum
        $|}
      )
  }
}
