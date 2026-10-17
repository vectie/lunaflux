///|
pub fn TextIoCudaSource::vocabulary_shard(
  self : Self,
) -> @parallel.VocabularyShard? {
  self.shard
}

///|
pub fn TextIoCudaSource::head_bytes(self : Self) -> Int64 {
  let (vocabulary, hidden, _, _, _) = self.plan.geometry()
  self.shard.map(s => s.columns().1).unwrap_or(vocabulary).to_int64() *
  hidden.to_int64() *
  2L
}

///|
pub fn TextIoCudaSource::local_logits_bytes(self : Self) -> Int64 {
  let (vocabulary, _, _, _, outputs) = self.plan.geometry()
  self.shard.map(s => s.columns().1).unwrap_or(vocabulary).to_int64() *
  outputs.to_int64() *
  4L
}

///|
pub fn TextIoCudaSource::workspace_bytes(self : Self) -> Int64 {
  self.plan.normalized_bytes() +
  self.local_logits_bytes() +
  (match self.shard {
    None => 0L
    Some(s) =>
      s.candidate_width().to_int64() * (s.group().1 + 1).to_int64() * 4L
  })
}

///|
fn sharded_greedy_source(
  plan : @precision.TextIoPrecision,
  shard : @parallel.VocabularyShard,
  symbol : String,
) -> String {
  let (_, _, _, rows, outputs) = plan.geometry()
  let (first, columns) = shard.columns()
  let (_, world) = shard.group()
  (
    $|extern "C" __global__ void \{symbol}_greedy(const int *counts,const int *output_count,const int *selected,const float *logits,float *candidates) {
    $| const int row=blockIdx.x,n=output_count[0],lane=threadIdx.x;
    $| __shared__ float values[256]; __shared__ int ids[256],invalid[256];
    $| const bool active=n>0&&n<=\{outputs}&&row<n;
    $| float best=-INFINITY; int token=-1,bad=!active;
    $| if(active) {
    $|   if(counts[3]<=0||counts[3]>\{rows}||selected[row]<0||selected[row]>=counts[3]) bad=1;
    $|   for(int word=lane;word<\{columns};word+=256) {
    $|     float value=logits[(unsigned long long)row*\{columns}+word];
    $|     if(!isfinite(value)) bad=1;
    $|     if(token<0||value>best) {best=value;token=word+\{first};}
    $|   }
    $| }
    $| values[lane]=best; ids[lane]=token; invalid[lane]=bad; __syncthreads();
    $| for(int span=128;span>0;span>>=1) { if(lane<span) {
    $|   int other=ids[lane+span]; float value=values[lane+span];
    $|   if(other>=0&&(ids[lane]<0||value>values[lane]||(value==values[lane]&&other<ids[lane]))) {values[lane]=value;ids[lane]=other;}
    $|   invalid[lane]|=invalid[lane+span];
    $| } __syncthreads(); }
    $| if(lane==0) {candidates[row*3]=values[0];candidates[row*3+1]=(float)ids[0];candidates[row*3+2]=(float)invalid[0];}
    $|}
    $|extern "C" __global__ void \{symbol}_greedy_join(const int *output_count,const float *all,int *tokens) {
    $| const int row=blockIdx.x,n=output_count[0]; if(threadIdx.x!=0||n<=0||n>\{outputs}||row>=n) return;
    $| float best=-INFINITY; int token=-1,bad=0;
    $| for(int rank=0;rank<\{world};++rank) {const float *p=all+(rank*\{outputs}+row)*3;int other=(int)p[1];
    $|   if(p[2]!=0.0f||!isfinite(p[0])||other<0) bad=1;
    $|   if(token<0||p[0]>best||(p[0]==best&&other<token)) {best=p[0];token=other;}
    $| } tokens[row]=bad?-1:token;
    $|}
  )
}
