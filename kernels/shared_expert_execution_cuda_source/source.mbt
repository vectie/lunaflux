///|
pub struct SharedExpertCudaSource {
  priv plan : @precision.SharedExpertPrecision
  priv symbol : String
  priv source : Bytes
}

///|
/// One CTA per row; gate sigmoid is rounded once and shared across channels.
pub fn shared_expert_source(
  plan : @precision.SharedExpertPrecision,
  symbol~ : String,
) -> SharedExpertCudaSource raise {
  let (width, rows) = plan.geometry()
  guard !symbol.is_empty() &&
    symbol.length() <= 128 &&
    symbol[0] is ('a'..<='z' | 'A'..<='Z' | '_') &&
    symbol.all(c => c is ('a'..<='z' | 'A'..<='Z' | '0'..<='9' | '_')) else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let sigmoid = @arithmetic.scalar_expression(plan.expression(Sigmoid), [
    "gate_logits[row]",
  ])
  let weight = @arithmetic.scalar_expression(plan.expression(Weight), [
    "shared[cell]", "gate",
  ])
  let combine = @arithmetic.scalar_expression(plan.expression(Combine), [
    "routed[cell]", "weighted",
  ])
  let text =
    $|#include <cuda_bf16.h>
    $|#include <stdint.h>
    $|#include <math.h>
    $|extern "C" __global__ void \{symbol}(const int *counts, const float *routed, const __nv_bfloat16 *shared, const __nv_bfloat16 *gate_logits, __nv_bfloat16 *output) {
    $|  const int row=blockIdx.x;
    $|  if(counts[3]<=0||counts[3]>\{rows}||row>=counts[3]) return;
    $|  __shared__ __nv_bfloat16 gate;
    $|  if(threadIdx.x==0) gate=\{sigmoid};
    $|  __syncthreads();
    $|  for(int column=threadIdx.x;column<\{width};column+=blockDim.x) {
    $|    const unsigned long long cell=(unsigned long long)row*\{width}ULL+column;
    $|    const __nv_bfloat16 weighted=\{weight};
    $|    output[cell]=\{combine};
    $|  }
    $|}
  { plan, symbol, source: @utf8.encode(text), }
}

///|
pub fn SharedExpertCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn SharedExpertCudaSource::plan(
  self : Self,
) -> @precision.SharedExpertPrecision {
  self.plan
}

///|
pub fn SharedExpertCudaSource::symbol(self : Self) -> String {
  self.symbol
}

///|
pub fn SharedExpertCudaSource::launch(self : Self) -> (Int32, Int32, Int32) {
  let (_, rows) = self.plan.geometry()
  (rows, 1, 128)
}

///|
pub fn SharedExpertCudaSource::parameter_count(_self : Self) -> Int32 {
  5
}
