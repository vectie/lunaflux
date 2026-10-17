///|
fn parallel_head_source(
  plan : @precision.TextIoPrecision,
  symbol : String,
  vocabulary_shard? : @parallel.VocabularyShard,
) -> String raise @precision.PrecisionPlanError {
  let (vocabulary, hidden, _, _, outputs) = plan.geometry()
  let vocabulary = vocabulary_shard
    .map(s => s.columns().1)
    .unwrap_or(vocabulary)
  let publish = @arithmetic.scalar_expression(plan.expression(Logits), ["value"])
  let body = @tiled.shape_decode_body(
    inner=hidden,
    columns=vocabulary,
    maximum_rows=outputs,
    left="__bfloat162float(hidden[(unsigned long long)row*\{hidden}+inner])",
    right="__bfloat162float(weight[(unsigned long long)column*\{hidden}+inner])",
    publish="logits[(unsigned long long)row*\{vocabulary}+column]=\{publish};",
  )
  (
    $|extern "C" __global__ void \{symbol}_head(const int *output_count,const __nv_bfloat16 *hidden,const __nv_bfloat16 *weight,float *logits) {
    $| const int n=output_count[0]; if(n<=0||n>\{outputs}) return;
    $| for(int row=0;row<n;++row) { \{body} }
    $|}
  )
}

///|
fn parallel_greedy_source(
  plan : @precision.TextIoPrecision,
  symbol : String,
) -> String {
  let (vocabulary, _, _, rows, outputs) = plan.geometry()
  let partition = try! @sampling.plan_greedy_segments(
    vocabulary_size=vocabulary,
    maximum_rows=outputs,
    segment_width=(vocabulary + 255) / 256,
  )
  let (width, segments) = partition.partition()
  (
    $|extern "C" __global__ void \{symbol}_greedy(const int *counts,const int *output_count,const int *selected,const float *logits,int *tokens) {
    $| const int row=blockIdx.x,n=output_count[0],lane=threadIdx.x;
    $| if(n<=0||n>\{outputs}||row>=n) return;
    $| if(counts[3]<=0||counts[3]>\{rows}||selected[row]<0||selected[row]>=counts[3]) {if(lane==0) tokens[row]=-1;return;}
    $| __shared__ float values[256]; __shared__ int ids[256],invalid[256];
    $| float best=-INFINITY; int token=-1,bad=0;
    $| if(lane<\{segments}) for(int word=lane*\{width};word<(lane+1)*\{width}&&word<\{vocabulary};++word) {
    $|   float value=logits[(unsigned long long)row*\{vocabulary}+word];
    $|   if(!isfinite(value)) bad=1;
    $|   if(token<0||value>best) {best=value;token=word;}
    $| }
    $| values[lane]=best; ids[lane]=token; invalid[lane]=bad; __syncthreads();
    $| for(int span=128;span>0;span>>=1) { if(lane<span) {
    $|   int other=ids[lane+span]; float value=values[lane+span];
    $|   if(other>=0&&(ids[lane]<0||value>values[lane]||(value==values[lane]&&other<ids[lane]))) {values[lane]=value;ids[lane]=other;}
    $|   invalid[lane]|=invalid[lane+span];
    $| } __syncthreads(); }
    $| if(lane==0) tokens[row]=invalid[0]?-1:ids[0];
    $|}
  )
}
