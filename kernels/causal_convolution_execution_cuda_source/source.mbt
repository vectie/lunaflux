///|
pub struct CausalConvolutionCudaSource {
  priv plan : @precision.CausalConvolutionPrecision
  priv symbol : String
  priv source : Bytes
}

///|
/// Each lane folds one channel across a sequence. Exclusive request slots,
/// CSR intervals and reset flags have the same meaning as the delta kernel.
/// In-place history is loaded once, shifted locally and committed after the
/// sequence. No sequence search, inter-CTA publication or per-token state copy.
pub fn causal_convolution_source(
  plan : @precision.CausalConvolutionPrecision,
  symbol~ : String,
) -> CausalConvolutionCudaSource raise {
  let (channels, kernel) = plan.geometry()
  let (rows, sequences, slots) = plan.capacity()
  guard kernel <= 32 &&
    sequences <= 65535 &&
    !symbol.is_empty() &&
    symbol.length() <= 128 &&
    symbol[0] is ('a'..='z' | 'A'..='Z' | '_') &&
    symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_')) else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let history = kernel - 1
  let text =
    $|#include <cuda_bf16.h>
    $|#include <stdint.h>
    $|#include <math.h>
    $|extern "C" __global__ void \{symbol}(const int *counts, const int *offsets, const int *slots, const int *reset, const __nv_bfloat16 *input, const __nv_bfloat16 *weight, __nv_bfloat16 *state, __nv_bfloat16 *output) {
    $|  const int sequences=counts[2], rows=counts[3], sequence=blockIdx.y;
    $|  const unsigned long long channel=(unsigned long long)blockIdx.x*128ULL+threadIdx.x;
    $|  if(sequences<=0||sequences>\{sequences}||rows<=0||rows>\{rows}||sequence>=sequences||channel>=\{channels}ULL) return;
    $|  const int begin=offsets[sequence], end=offsets[sequence+1], slot=slots[sequence];
    $|  if(begin<0||end<=begin||end>rows||slot<0||slot>=\{slots}) return;
    $|  const unsigned long long state_base=((unsigned long long)slot*\{channels}ULL+channel)*\{history}ULL;
    $|  __nv_bfloat16 past[\{history}];
    $|  for(int lag=0;lag<\{history};++lag) past[lag]=reset[sequence] ? __float2bfloat16_rn(0.0f) : state[state_base+lag];
    $|  for(int row=begin;row<end;++row) {
    $|    const unsigned long long cell=(unsigned long long)row*\{channels}ULL+channel;
    $|    const __nv_bfloat16 current=input[cell]; float sum=0.0f;
    $|    for(int lag=0;lag<\{history};++lag) sum=__fadd_rn(sum,__fmul_rn(__bfloat162float(past[lag]),__bfloat162float(weight[channel*\{kernel}ULL+lag])));
    $|    sum=__fadd_rn(sum,__fmul_rn(__bfloat162float(current),__bfloat162float(weight[channel*\{kernel}ULL+\{history}ULL])));
    $|    output[cell]=__float2bfloat16_rn(__fdiv_rn(sum,__fadd_rn(1.0f,expf(-sum))));
    $|    for(int lag=0;lag<\{history - 1};++lag) past[lag]=past[lag+1]; past[\{history - 1}]=current;
    $|  }
    $|  for(int lag=0;lag<\{history};++lag) state[state_base+lag]=past[lag];
    $|}
  { plan, symbol, source: @utf8.encode(text), }
}

///|
pub fn CausalConvolutionCudaSource::plan(
  self : Self,
) -> @precision.CausalConvolutionPrecision {
  self.plan
}

///|
pub fn CausalConvolutionCudaSource::symbol(self : Self) -> String {
  self.symbol
}

///|
pub fn CausalConvolutionCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn CausalConvolutionCudaSource::launch(self : Self) -> (Int, Int, Int) {
  let (channels, _) = self.plan.geometry()
  let (_, sequences, _) = self.plan.capacity()
  (((channels.to_int64() + 127L) / 128L).to_int(), sequences, 128)
}
