///|
pub struct GatedDeltaControlCudaSource {
  priv plan : @precision.GatedDeltaControlPrecision
  priv symbol : String
  priv source : Bytes
}

///|
pub fn gated_delta_control_source(
  plan : @precision.GatedDeltaControlPrecision,
  symbol~ : String,
) -> GatedDeltaControlCudaSource raise {
  let (heads, rows) = plan.geometry()
  guard !symbol.is_empty() &&
    symbol.length() <= 128 &&
    symbol[0] is ('a'..<='z' | 'A'..<='Z' | '_') &&
    symbol.all(c => c is ('a'..<='z' | 'A'..<='Z' | '0'..<='9' | '_')) else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let beta = @arithmetic.scalar_expression(plan.expression(Beta), ["b[index]"])
  let decay = @arithmetic.scalar_expression(plan.expression(LogDecay), [
    "a[index]", "a_log[head]", "dt_bias[head]",
  ])
  let text =
    $|#include <cuda_bf16.h>
    $|#include <stdint.h>
    $|#include <math.h>
    $|extern "C" __global__ void \{symbol}(const int *counts, const __nv_bfloat16 *a, const __nv_bfloat16 *b, const __nv_bfloat16 *a_log, const __nv_bfloat16 *dt_bias, float *log_decay, __nv_bfloat16 *beta) {
    $|  const int row=blockIdx.x;
    $|  if(counts[3]<=0||counts[3]>\{rows}||row>=counts[3]) return;
    $|  for(int head=threadIdx.x;head<\{heads};head+=blockDim.x) {
    $|    const unsigned long long index=(unsigned long long)row*\{heads}ULL+head;
    $|    log_decay[index]=\{decay};
    $|    beta[index]=\{beta};
    $|  }
    $|}
  { plan, symbol, source: @utf8.encode(text), }
}

///|
pub fn GatedDeltaControlCudaSource::source_bytes(
  self : GatedDeltaControlCudaSource,
) -> Bytes {
  self.source
}

///|
pub fn GatedDeltaControlCudaSource::plan(
  self : GatedDeltaControlCudaSource,
) -> @precision.GatedDeltaControlPrecision {
  self.plan
}

///|
pub fn GatedDeltaControlCudaSource::symbol(
  self : GatedDeltaControlCudaSource,
) -> String {
  self.symbol
}

///|
pub fn GatedDeltaControlCudaSource::launch(
  self : GatedDeltaControlCudaSource,
) -> (Int32, Int32, Int32) {
  let (_, rows) = self.plan.geometry()
  (rows, 1, 128)
}

///|
pub fn GatedDeltaControlCudaSource::parameter_count(
  _self : GatedDeltaControlCudaSource,
) -> Int32 {
  7
}
