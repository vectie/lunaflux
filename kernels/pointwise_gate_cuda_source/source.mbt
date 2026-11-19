///|
pub struct PointwiseGateCudaSource {
  priv plan : @precision.PointwiseGatePrecision
  priv symbol : String
  priv source : Bytes
}

///|
pub fn pointwise_gate_source(
  plan : @precision.PointwiseGatePrecision,
  symbol~ : String,
) -> PointwiseGateCudaSource raise {
  let (width, vectors, rows) = plan.geometry()
  let (vector_stride, row_stride, offset) = plan.gate_layout()
  let row_width = width.to_int64() * vectors.to_int64()
  guard !symbol.is_empty() &&
    symbol.length() <= 128 &&
    symbol[0] is ('a'..<='z' | 'A'..<='Z' | '_') &&
    symbol.all(c => c is ('a'..<='z' | 'A'..<='Z' | '0'..<='9' | '_')) else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let sigmoid = @arithmetic.scalar_expression(
    @precision.Bf16GateExpression::Sigmoid.program(),
    ["gate[gate_index]"],
  )
  let product = @arithmetic.scalar_expression(
    @precision.Bf16GateExpression::Product.program(),
    ["input[cell]", "activated"],
  )
  let text =
    $|#include <cuda_bf16.h>
    $|#include <stdint.h>
    $|#include <math.h>
    $|extern "C" __global__ void \{symbol}(const int *counts, const __nv_bfloat16 *input, const __nv_bfloat16 *gate, __nv_bfloat16 *output) {
    $|  const int row=blockIdx.x;
    $|  if(counts[3]<=0||counts[3]>\{rows}||row>=counts[3]) return;
    $|  for(unsigned long long channel=threadIdx.x;channel<\{row_width}ULL;channel+=blockDim.x) {
    $|    const unsigned long long cell=(unsigned long long)row*\{row_width}ULL+channel;
    $|    const unsigned long long gate_index=(unsigned long long)row*\{row_stride}ULL+(channel/\{width})*\{vector_stride}ULL+\{offset}ULL+channel%\{width};
    $|    const __nv_bfloat16 activated=\{sigmoid};
    $|    output[cell]=\{product};
    $|  }
    $|}
  { plan, symbol, source: @utf8.encode(text), }
}

///|
pub fn PointwiseGateCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn PointwiseGateCudaSource::symbol(self : Self) -> String {
  self.symbol
}

///|
pub fn PointwiseGateCudaSource::plan(
  self : Self,
) -> @precision.PointwiseGatePrecision {
  self.plan
}

///|
pub fn PointwiseGateCudaSource::launch(self : Self) -> (Int32, Int32, Int32) {
  let (_, _, rows) = self.plan.geometry()
  (rows, 1, 128)
}

///|
pub fn PointwiseGateCudaSource::parameter_count(_self : Self) -> Int32 {
  4
}
