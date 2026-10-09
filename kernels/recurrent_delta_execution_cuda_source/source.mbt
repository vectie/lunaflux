///|
pub struct RecurrentDeltaCudaSource {
  priv plan : @precision.RecurrentDeltaPrecision
  priv symbol : String
  priv source : Bytes
  priv block : Int
}

///|
/// One CTA owns a sequence/head and each value lane owns an ordered key fold.
/// Lane-private state survives across the sequence's token recurrence.
/// Slots must be exclusive within a frame; the scheduler owns that guarantee.
pub fn recurrent_delta_source(
  plan : @precision.RecurrentDeltaPrecision,
  symbol~ : String,
) -> RecurrentDeltaCudaSource raise {
  let (heads, keys, values) = plan.geometry()
  let (rows, sequences, slots) = plan.capacity()
  guard keys <= 256 &&
    values <= 256 &&
    heads <= 65535 &&
    !symbol.is_empty() &&
    symbol.length() <= 128 &&
    symbol[0] is ('a'..='z' | 'A'..='Z' | '_') &&
    symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_')) else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let block = ((if keys > values { keys } else { values }) + 31) / 32 * 32
  let epsilon = @literal.canonical_cuda_float_literal(
    plan.normalization_epsilon(),
  )
  let text =
    $|#include <cuda_bf16.h>
    $|#include <stdint.h>
    $|#include <math.h>
    $|extern "C" __global__ void \{symbol}(const int *counts, const int *offsets, const int *slots, const int *reset, const __nv_bfloat16 *query, const __nv_bfloat16 *key, const __nv_bfloat16 *value, const float *log_decay, const __nv_bfloat16 *beta, float *state, __nv_bfloat16 *output) {
    $|  const int sequences=counts[2], rows=counts[3], sequence=blockIdx.x, head=blockIdx.y, lane=threadIdx.x;
    $|  if(sequences<=0||sequences>\{sequences}||rows<=0||rows>\{rows}||sequence>=sequences||head>=\{heads}) return;
    $|  const int begin=offsets[sequence], end=offsets[sequence+1], slot=slots[sequence];
    $|  if(begin<0||end<=begin||end>rows||slot<0||slot>=\{slots}) return;
    $|  const unsigned long long state_base=((unsigned long long)slot*\{heads}ULL+head)*\{keys}ULL*\{values}ULL;
    $|  float memory_state[\{keys}];
    $|  if(lane<\{values}) for(int k=0;k<\{keys};++k) memory_state[k]=reset[sequence] ? 0.0f : state[state_base+(unsigned long long)k*\{values}ULL+lane];
    $|  __shared__ float q[\{keys}], kn[\{keys}], decay[\{keys}], qinv, kinv, b;
    $|  const float qscale=__fdiv_rn(1.0f,sqrtf(\{keys}.0f));
    $|  for(int row=begin;row<end;++row) {
    $|    const unsigned long long qbase=((unsigned long long)row*\{heads}ULL+head)*\{keys}ULL;
    $|    const unsigned long long vbase=((unsigned long long)row*\{heads}ULL+head)*\{values}ULL;
    $|    if(lane<\{keys}) { q[lane]=__bfloat162float(query[qbase+lane]); kn[lane]=__bfloat162float(key[qbase+lane]); decay[lane]=expf(log_decay[qbase+lane]); }
    $|    __syncthreads();
    $|    if(lane==0) {
    $|      float qs=0.0f,ks=0.0f;
    $|      for(int k=0;k<\{keys};++k) { qs=__fadd_rn(qs,__fmul_rn(q[k],q[k])); ks=__fadd_rn(ks,__fmul_rn(kn[k],kn[k])); }
    $|      qinv=__fdiv_rn(1.0f,sqrtf(__fadd_rn(qs,\{epsilon}))); kinv=__fdiv_rn(1.0f,sqrtf(__fadd_rn(ks,\{epsilon})));
    $|      b=__bfloat162float(beta[(unsigned long long)row*\{heads}ULL+head]);
    $|    }
    $|    __syncthreads();
    $|    if(lane<\{keys}) { q[lane]=__fmul_rn(__fmul_rn(q[lane],qinv),qscale); kn[lane]=__fmul_rn(kn[lane],kinv); }
    $|    __syncthreads();
    $|    if(lane<\{values}) {
    $|      float read=0.0f;
    $|      for(int k=0;k<\{keys};++k) { memory_state[k]=__fmul_rn(memory_state[k],decay[k]); read=__fadd_rn(read,__fmul_rn(memory_state[k],kn[k])); }
    $|      const float delta=__fmul_rn(__fsub_rn(__bfloat162float(value[vbase+lane]),read),b);
    $|      float result=0.0f;
    $|      for(int k=0;k<\{keys};++k) { memory_state[k]=__fadd_rn(memory_state[k],__fmul_rn(kn[k],delta)); result=__fadd_rn(result,__fmul_rn(memory_state[k],q[k])); }
    $|      output[vbase+lane]=__float2bfloat16_rn(result);
    $|    }
    $|    __syncthreads();
    $|  }
    $|  if(lane<\{values}) for(int k=0;k<\{keys};++k) state[state_base+(unsigned long long)k*\{values}ULL+lane]=memory_state[k];
    $|}
  { plan, symbol, source: @utf8.encode(text), block, }
}

///|
pub fn RecurrentDeltaCudaSource::plan(
  self : Self,
) -> @precision.RecurrentDeltaPrecision {
  self.plan
}

///|
pub fn RecurrentDeltaCudaSource::symbol(self : Self) -> String {
  self.symbol
}

///|
pub fn RecurrentDeltaCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn RecurrentDeltaCudaSource::launch(self : Self) -> (Int, Int, Int) {
  let (_, sequences, _) = self.plan.capacity()
  let (heads, _, _) = self.plan.geometry()
  (sequences, heads, self.block)
}
