///|
pub struct RmsNormCudaSource {
  priv plan : @precision.RmsNormPrecision
  priv symbol : String
  priv source : Bytes
}

///|
/// Ordered FP32 square fold, explicit model stage rounds, packed BF16 output.
/// One CTA owns one vector. Gate vectors, when present, use packed output layout.
pub fn rms_norm_source(
  plan : @precision.RmsNormPrecision,
  symbol~ : String,
) -> RmsNormCudaSource raise {
  let (width, vectors, rows) = plan.geometry()
  let (vector_stride, row_stride) = plan.input_layout()
  guard vectors <= 65535 &&
    !symbol.is_empty() &&
    symbol.length() <= 128 &&
    symbol[0] is ('a'..<='z' | 'A'..<='Z' | '_') &&
    symbol.all(c => c is ('a'..<='z' | 'A'..<='Z' | '0'..<='9' | '_')) else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let epsilon = @literal.canonical_cuda_float_literal(plan.epsilon())
  let fold = plan.square_fold()
  let square = @arithmetic.scalar_expression(fold.update, ["sum", "x"])
  let inverse = @arithmetic.scalar_expression(plan.expression(Inverse), [
    "sum",
    "\{width}.0f",
    epsilon,
  ])
  let normalized = @arithmetic.scalar_expression(plan.expression(Normalize), [
    "x", "inverse",
  ])
  let weighted = @arithmetic.scalar_expression(plan.expression(Weight), [
    "normalized", "weight[column]",
  ])
  let (gate_argument, published) = match plan.mode() {
    Bf16NormBf16WeightSiluGate =>
      (
        ", const __nv_bfloat16 *gate",
        @arithmetic.scalar_expression(plan.expression(Gate), [
          "weighted", "gate[out_base+column]",
        ]),
      )
    _ =>
      (
        "",
        @arithmetic.scalar_expression(plan.expression(Publish), ["weighted"]),
      )
  }
  let text =
    $|#include <cuda_bf16.h>
    $|#include <stdint.h>
    $|#include <math.h>
    $|extern "C" __global__ void \{symbol}(const int *counts, const __nv_bfloat16 *input, const __nv_bfloat16 *weight\{gate_argument}, __nv_bfloat16 *output) {
    $|  const int row=blockIdx.x, vector=blockIdx.y;
    $|  if(counts[3]<=0||counts[3]>\{rows}||row>=counts[3]||vector>=\{vectors}) return;
    $|  const unsigned long long input_base=(unsigned long long)row*\{row_stride}ULL+(unsigned long long)vector*\{vector_stride}ULL;
    $|  const unsigned long long out_base=((unsigned long long)row*\{vectors}ULL+vector)*\{width}ULL;
    $|  __shared__ float inverse;
    $|  if(threadIdx.x==0) {
    $|    float sum=0.0f;
    $|    for(int column=0;column<\{width};++column) { const float x=__bfloat162float(input[input_base+column]); sum=\{square}; }
    $|    inverse=\{inverse};
    $|  }
    $|  __syncthreads();
    $|  for(int column=threadIdx.x;column<\{width};column+=blockDim.x) {
    $|    const float x=__bfloat162float(input[input_base+column]);
    $|    const float normalized=\{normalized};
    $|    const float weighted=\{weighted};
    $|    output[out_base+column]=\{published};
    $|  }
    $|}
  { plan, symbol, source: @utf8.encode(text), }
}

///|
pub fn RmsNormCudaSource::source_bytes(self : RmsNormCudaSource) -> Bytes {
  self.source
}

///|
pub fn RmsNormCudaSource::plan(
  self : RmsNormCudaSource,
) -> @precision.RmsNormPrecision {
  self.plan
}

///|
pub fn RmsNormCudaSource::symbol(self : RmsNormCudaSource) -> String {
  self.symbol
}

///|
pub fn RmsNormCudaSource::launch(
  self : RmsNormCudaSource,
) -> (Int32, Int32, Int32) {
  let (_, vectors, rows) = self.plan.geometry()
  (rows, vectors, 128)
}

///|
pub fn RmsNormCudaSource::parameter_count(self : RmsNormCudaSource) -> Int32 {
  if self.plan.mode() is Bf16NormBf16WeightSiluGate {
    5
  } else {
    4
  }
}
