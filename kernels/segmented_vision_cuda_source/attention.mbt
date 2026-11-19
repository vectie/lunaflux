///|
pub struct AttentionSource {
  priv plan : @precision.SegmentedAttentionPrecision
  priv symbol : String
  priv source : Bytes
}

///|
/// AOT bounded packed attention. counts[3] is the live row count. Each CTA
/// owns one query/head; 128 threads compute keys, then value lanes. Integer
/// spans are prepared by the model owner and can be replaced between images.
pub fn attention_source(
  plan : @precision.SegmentedAttentionPrecision,
  symbol~ : String,
) -> AttentionSource raise {
  let (rows, segment, heads, width) = plan.geometry()
  validate(symbol, rows, heads * width)
  guard segment <= 8192 && width <= 128 else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let dot = @arithmetic.scalar_expression(plan.expression(DotUpdate), [
    "dot", "q[qbase+d]", "k[kbase+d]",
  ])
  let score = @arithmetic.scalar_expression(plan.expression(ScaledScore), [
    "dot", "scale",
  ])
  let probability = @arithmetic.scalar_expression(
    plan.expression(ProbabilityStorage),
    ["expf(scores[key]-maximum)", "denominator"],
  )
  let update = @arithmetic.scalar_expression(plan.expression(ValueUpdate), [
    "sum", "scores[key]", "v[vbase+lane]",
  ])
  let publish = @arithmetic.scalar_expression(plan.expression(Publish), ["sum"])
  let scale = Float::from_double(1.0 / width.to_double().sqrt()).to_string() +
    "f"
  let source =
    $|#include <cuda_bf16.h>
    $|#include <stdint.h>
    $|#include <math.h>
    $|extern "C" __global__ void \{symbol}(const int32_t* counts,const int32_t* starts,const int32_t* ends,const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out) {
    $| const int row=blockIdx.x,head=blockIdx.y,lane=threadIdx.x;
    $| const int live=counts[3];if(live<=0||live>\{rows}||row>=live||head>=\{heads})return;
    $| const int first=starts[row],last=ends[row],length=last-first;
    $| if(first<0||first>row||last<=row||last>live||length>\{segment})return;
    $| const unsigned long long qbase=((unsigned long long)row*\{heads}+head)*\{width};
    $| const float scale=\{scale};
    $| __shared__ float scores[\{segment}],reduce[128];
    $| float local_max=-INFINITY;
    $| for(int key=lane;key<length;key+=128){
    $|  const unsigned long long kbase=((unsigned long long)(first+key)*\{heads}+head)*\{width};float dot=0.0f;
    $|  for(int d=0;d<\{width};++d)dot=\{dot};
    $|  const float score=\{score};scores[key]=score;local_max=fmaxf(local_max,score);
    $| }
    $| reduce[lane]=local_max;__syncthreads();
    $| for(int step=64;step;step>>=1){if(lane<step)reduce[lane]=fmaxf(reduce[lane],reduce[lane+step]);__syncthreads();}
    $| const float maximum=reduce[0];float local_sum=0.0f;
    $| for(int key=lane;key<length;key+=128)local_sum=__fadd_rn(local_sum,expf(scores[key]-maximum));
    $| __syncthreads();reduce[lane]=local_sum;__syncthreads();
    $| for(int step=64;step;step>>=1){if(lane<step)reduce[lane]=__fadd_rn(reduce[lane],reduce[lane+step]);__syncthreads();}
    $| const float denominator=reduce[0];
    $| for(int key=lane;key<length;key+=128)scores[key]=\{probability};
    $| __syncthreads();
    $| if(lane<\{width}){float sum=0.0f;for(int key=0;key<length;++key){
    $|  const unsigned long long vbase=((unsigned long long)(first+key)*\{heads}+head)*\{width};sum=\{update};
    $| }out[qbase+lane]=\{publish};}
    $|}
  { plan, symbol, source: @utf8.encode(source), }
}

///|
pub fn AttentionSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn AttentionSource::symbol(self : Self) -> String {
  self.symbol
}

///|
pub fn AttentionSource::plan(
  self : Self,
) -> @precision.SegmentedAttentionPrecision {
  self.plan
}

///|
pub fn AttentionSource::launch(self : Self) -> (Int32, Int32, Int32) {
  let (rows, _, heads, _) = self.plan.geometry()
  (rows, heads, 128)
}

///|
pub fn AttentionSource::shared_bytes(self : Self) -> Int32 {
  let (_, segment, _, _) = self.plan.geometry()
  segment * 4 + 128 * 4
}

///|
pub fn AttentionSource::parameter_count(_self : Self) -> Int32 {
  7
}
