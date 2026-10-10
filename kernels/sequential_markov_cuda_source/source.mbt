///|
pub struct SequentialMarkovCudaSource {
  priv plan : @precision.SequentialMarkovPrecision
  priv prefix : String
  priv bytes : Bytes
}

///|
/// Four ordered kernels per row preserve the read-after-sample dependency.
/// Hardware distribution and stateless RNG lowering stay in this backend.
pub fn sequential_markov_source(
  plan : @precision.SequentialMarkovPrecision,
  symbol_prefix~ : String,
) -> SequentialMarkovCudaSource raise {
  guard !symbol_prefix.is_empty() &&
    symbol_prefix.length() <= 80 &&
    symbol_prefix[0] is ('a'..='z' | 'A'..='Z' | '_') &&
    symbol_prefix.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_')) else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let (v, h, k, rows) = plan.geometry()
  let temperature = @literal.canonical_cuda_float_literal(
    plan.temperature().max(1.0e-5),
  )
  let score = if plan.temperature() == 0.0 {
    "value"
  } else {
    symbol_prefix + "_draw(value,rng[0],rng[1],row,word)"
  }
  let source =
    $|#include <cuda_bf16.h>
    $|#include <stdint.h>
    $|#include <math.h>
    $|static __device__ __forceinline__ uint64_t \{symbol_prefix}_mix(uint64_t x) {
    $|  x += 0x9e3779b97f4a7c15ULL;
    $|  x = (x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    $|  x = (x^(x>>27))*0x94d049bb133111ebULL; return x^(x>>31);
    $|}
    $|static __device__ __forceinline__ float \{symbol_prefix}_draw(float value,uint64_t seed,uint64_t sequence,int row,int word) {
    $|  const uint64_t bits=\{symbol_prefix}_mix(seed^\{symbol_prefix}_mix(sequence)^\{symbol_prefix}_mix((uint64_t)row*\{v}ULL+word));
    $|  const float uniform=__fdiv_rn((float)((bits>>41)+1ULL),8388609.0f);
    $|  return __fsub_rn(__fdiv_rn(value,\{temperature}),logf(-logf(uniform)));
    $|}
    $|extern "C" __global__ void \{symbol_prefix}_gather(const int32_t* count,const int32_t* tokens,const __nv_bfloat16* table,__nv_bfloat16* embeddings,const int32_t* row_ref) {
    $|  const int row=row_ref[0];
    $|  if(count[0]<=0 || count[0]>\{rows} || row>=count[0]) return;
    $|  const int token=tokens[row];
    $|  for(int c=threadIdx.x;c<\{k};c+=blockDim.x)
    $|    embeddings[(uint64_t)row*\{k}+c]=(token>=0 && token<\{v})?table[(uint64_t)token*\{k}+c]:__float2bfloat16_rn(0.0f);
    $|}
    $|extern "C" __global__ void \{symbol_prefix}_bias(const int32_t* count,const int32_t* tokens,const __nv_bfloat16* embeddings,const __nv_bfloat16* weight,float* logits,const int32_t* row_ref) {
    $|  const int row=row_ref[0];
    $|  const int word=blockIdx.x*blockDim.x+threadIdx.x;
    $|  if(count[0]<=0 || count[0]>\{rows} || row>=count[0] || word>=\{v}) return;
    $|  float sum=0.0f;
    $|  for(int c=0;c<\{k};++c) sum=__fadd_rn(sum,__fmul_rn(__bfloat162float(embeddings[(uint64_t)row*\{k}+c]),__bfloat162float(weight[(uint64_t)word*\{k}+c])));
    $|  const uint64_t index=(uint64_t)row*\{v}+word;
    $|  logits[index]=(tokens[row]>=0 && tokens[row]<\{v})?__fadd_rn(logits[index],sum):NAN;
    $|}
    $|extern "C" __global__ void \{symbol_prefix}_sample(const int32_t* count,const float* logits,const uint64_t* rng,int32_t* tokens,const int32_t* row_ref) {
    $|  const int row=row_ref[0];
    $|  if(count[0]<=0 || count[0]>\{rows} || row>=count[0]) return;
    $|  __shared__ float values[256]; __shared__ int ids[256],bad[256];
    $|  const int t=threadIdx.x; float best=-INFINITY; int id=-1,invalid=0;
    $|  for(int word=t;word<\{v};word+=256) {
    $|    const float value=logits[(uint64_t)row*\{v}+word];
    $|    invalid|=!isfinite(value); const float candidate=\{score};
    $|    if(id<0 || candidate>best || (candidate==best && word<id)) {best=candidate;id=word;}
    $|  }
    $|  values[t]=best; ids[t]=id; bad[t]=invalid; __syncthreads();
    $|  for(int width=128;width>0;width>>=1) {
    $|    if(t<width) {
    $|      const int other=ids[t+width]; const float value=values[t+width];
    $|      if(other>=0 && (ids[t]<0 || value>values[t] || (value==values[t] && other<ids[t]))) {values[t]=value;ids[t]=other;}
    $|      bad[t]|=bad[t+width];
    $|    } __syncthreads();
    $|  }
    $|  if(t==0) tokens[row+1]=bad[0]?-1:ids[0];
    $|}
    $|extern "C" __global__ void \{symbol_prefix}_confidence(const int32_t* count,const int32_t* tokens,const __nv_bfloat16* hidden,const __nv_bfloat16* embeddings,const __nv_bfloat16* weight,float* confidence,const int32_t* row_ref) {
    $|  const int row=row_ref[0];
    $|  if(threadIdx.x!=0 || count[0]<=0 || count[0]>\{rows} || row>=count[0]) return;
    $|  float sum=0.0f;
    $|  for(int c=0;c<\{h};++c) sum=__fadd_rn(sum,__fmul_rn(__bfloat162float(hidden[(uint64_t)row*\{h}+c]),__bfloat162float(weight[c])));
    $|  for(int c=0;c<\{k};++c) sum=__fadd_rn(sum,__fmul_rn(__bfloat162float(embeddings[(uint64_t)row*\{k}+c]),__bfloat162float(weight[\{h}+c])));
    $|  confidence[row]=(tokens[row]>=0 && tokens[row]<\{v})?sum:NAN;
    $|}
  { plan, prefix: symbol_prefix, bytes: @utf8.encode(source), }
}

///|
pub fn SequentialMarkovCudaSource::plan(
  self : Self,
) -> @precision.SequentialMarkovPrecision {
  self.plan
}

///|
pub fn SequentialMarkovCudaSource::source_bytes(self : Self) -> Bytes {
  self.bytes
}

///|
pub fn SequentialMarkovCudaSource::symbol(
  self : Self,
  stage : Int,
) -> String raise {
  let suffix = match stage {
    0 => "_gather"
    1 => "_bias"
    2 => "_sample"
    3 => "_confidence"
    _ => raise @precision.PrecisionPlanError::IllegalPlacement
  }
  self.prefix + suffix
}

///|
pub fn SequentialMarkovCudaSource::launch(
  self : Self,
  stage : Int,
) -> (Int, Int) raise {
  let (v, _, _, _) = self.plan.geometry()
  match stage {
    0 | 2 => (1, 256)
    1 => ((v - 1) / 256 + 1, 256)
    3 => (1, 1)
    _ => raise @precision.PrecisionPlanError::IllegalPlacement
  }
}
