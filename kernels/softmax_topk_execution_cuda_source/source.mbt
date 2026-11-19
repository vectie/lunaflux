///|
pub struct SoftmaxTopKCudaSource {
  priv plan : @precision.SoftmaxTopKPrecision
  priv symbol : String
  priv source : Bytes
}

///|
/// Initial ordered implementation, one CTA per row. Projection remains a
/// separate BF16 matrix operator. No device allocation or host selection.
pub fn softmax_topk_source(
  plan : @precision.SoftmaxTopKPrecision,
  symbol~ : String,
) -> SoftmaxTopKCudaSource raise {
  let (experts, selected, rows) = plan.geometry()
  guard experts <= 4096 &&
    !symbol.is_empty() &&
    symbol.length() <= 128 &&
    symbol[0] is ('a'..<='z' | 'A'..<='Z' | '_') &&
    symbol.all(c => c is ('a'..<='z' | 'A'..<='Z' | '0'..<='9' | '_')) else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let exponential = @arithmetic.scalar_expression(
    plan.expression(ShiftedExponential),
    ["logits[base+expert]", "maximum"],
  )
  let sum = @arithmetic.scalar_expression(plan.expression(Sum), [
    "total", "probabilities[expert]",
  ])
  let probability = @arithmetic.scalar_expression(
    plan.expression(Probability),
    ["probabilities[expert]", "total"],
  )
  let selected_sum = @arithmetic.scalar_expression(plan.expression(Sum), [
    "selected_total", "top[rank]",
  ])
  let weight = @arithmetic.scalar_expression(plan.expression(SelectedWeight), [
    "top[rank]", "selected_total",
  ])
  let weight_type = if plan.weight_storage() is Bf16Weights {
    "__nv_bfloat16"
  } else {
    "float"
  }
  let text =
    $|#include <cuda_bf16.h>
    $|#include <stdint.h>
    $|#include <math.h>
    $|extern "C" __global__ void \{symbol}(const int *counts, const __nv_bfloat16 *logits, int *indices, \{weight_type} *weights) {
    $|  const int row=blockIdx.x;
    $|  if(threadIdx.x!=0||counts[3]<=0||counts[3]>\{rows}||row>=counts[3]) return;
    $|  const unsigned long long base=(unsigned long long)row*\{experts}ULL, out=(unsigned long long)row*\{selected}ULL;
    $|  float maximum=-INFINITY;
    $|  for(int expert=0;expert<\{experts};++expert) maximum=fmaxf(maximum,__bfloat162float(logits[base+expert]));
    $|  float probabilities[\{experts}], top[\{selected}];
    $|  float total=0.0f;
    $|  for(int expert=0;expert<\{experts};++expert) { probabilities[expert]=\{exponential}; total=\{sum}; }
    $|  for(int expert=0;expert<\{experts};++expert) probabilities[expert]=\{probability};
    $|  for(int rank=0;rank<\{selected};++rank) {
    $|    int best=0;
    $|    for(int expert=1;expert<\{experts};++expert) if(probabilities[expert]>probabilities[best]) best=expert;
    $|    indices[out+rank]=best; top[rank]=probabilities[best]; probabilities[best]=-1.0f;
    $|  }
    $|  float selected_total=0.0f;
    $|  for(int rank=0;rank<\{selected};++rank) selected_total=\{selected_sum};
    $|  for(int rank=0;rank<\{selected};++rank) weights[out+rank]=\{weight};
    $|}
  { plan, symbol, source: @utf8.encode(text), }
}

///|
pub fn SoftmaxTopKCudaSource::source_bytes(
  self : SoftmaxTopKCudaSource,
) -> Bytes {
  self.source
}

///|
pub fn SoftmaxTopKCudaSource::plan(
  self : SoftmaxTopKCudaSource,
) -> @precision.SoftmaxTopKPrecision {
  self.plan
}

///|
pub fn SoftmaxTopKCudaSource::symbol(self : SoftmaxTopKCudaSource) -> String {
  self.symbol
}

///|
pub fn SoftmaxTopKCudaSource::launch(
  self : SoftmaxTopKCudaSource,
) -> (Int32, Int32, Int32) {
  let (_, _, rows) = self.plan.geometry()
  (rows, 1, 32)
}

///|
pub fn SoftmaxTopKCudaSource::parameter_count(
  _self : SoftmaxTopKCudaSource,
) -> Int32 {
  4
}
