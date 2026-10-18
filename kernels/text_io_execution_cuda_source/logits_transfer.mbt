///|
/// Pure transport of the model's F32 logit values. No random choice or reference
/// comparison enters this lowering. Invalid selected rows retain a NaN error,
/// never a fabricated greedy result; inactive transport rows are defined zero.
fn sharded_logits_source(
  plan : @precision.TextIoPrecision,
  shard : @parallel.VocabularyShard,
  symbol : String,
) -> String {
  let (vocabulary, _, _, rows, outputs) = plan.geometry()
  let (_, columns) = shard.columns()
  let (_, world) = shard.group()
  let padded = shard.padded_columns()
  (
    $|extern "C" __global__ void \{symbol}_logits_pack(const int *counts,const int *output_count,const int *selected,const float *logits,float *packed) {
    $| const int row=blockIdx.x,n=output_count[0]; if(row>=\{outputs}) return;
    $| const bool active=n>0&&n<=\{outputs}&&row<n;
    $| const bool valid=active&&counts[3]>0&&counts[3]<=\{rows}&&selected[row]>=0&&selected[row]<counts[3];
    $| for(int word=threadIdx.x;word<\{padded};word+=blockDim.x) {
    $|   float value=0.0f;
    $|   if(active&&word<\{columns}) value=valid?logits[(unsigned long long)row*\{columns}+word]:NAN;
    $|   packed[(unsigned long long)row*\{padded}+word]=value;
    $| }
    $|}
    $|extern "C" __global__ void \{symbol}_logits_join(const int *output_count,const float *all,float *logits) {
    $| const int row=blockIdx.x,n=output_count[0]; if(n<=0||n>\{outputs}||row>=n) return;
    $| for(int rank=0;rank<\{world};++rank) {
    $|   const int first=(int)((unsigned long long)\{vocabulary}*rank/\{world});
    $|   const int end=(int)((unsigned long long)\{vocabulary}*(rank+1)/\{world});
    $|   for(int word=threadIdx.x;word<end-first;word+=blockDim.x)
    $|     logits[(unsigned long long)row*\{vocabulary}+first+word]=all[((unsigned long long)rank*\{outputs}+row)*\{padded}+word];
    $| }
    $|}
  )
}
