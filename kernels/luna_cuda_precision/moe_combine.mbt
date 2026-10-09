///|
pub struct MoeCombineSource {
  priv text : String
  priv symbol : String
  priv launch : ConversionLaunch
  priv plan : @precision.MoeCombinePrecision
}

///|
pub fn MoeCombineSource::source(self : Self) -> String {
  self.text
}

///|
pub fn MoeCombineSource::symbol(self : Self) -> String {
  self.symbol
}

///|
pub fn MoeCombineSource::launch(self : Self) -> ConversionLaunch {
  self.launch
}

///|
pub fn MoeCombineSource::plan(self : Self) -> @precision.MoeCombinePrecision {
  self.plan
}

///|
pub fn moe_combine_source(
  plan : @precision.MoeCombinePrecision,
  symbol~ : String,
) -> MoeCombineSource raise LoweringError {
  guard valid_symbol(symbol) else { raise InvalidSymbol }
  let (rows, hidden) = plan.geometry()
  let cells = rows.to_int64() * hidden.to_int64()
  let routed = match plan.rounding() {
    SumThenBf16 => "routed[i]"
    Bf16RoutedThenSum => "__bfloat162float(__float2bfloat16_rn(routed[i]))"
  }
  let text =
    $|#include <cuda_bf16.h>
    $|extern "C" __global__ void \{symbol}(const int *counts, const float *routed, const float *shared, __nv_bfloat16 *output) {
    $|  const int live=counts[3]; if(live<=0||live>\{rows}) return;
    $|  const unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    $|  if(i>=(unsigned long long)live*\{hidden}ULL) return;
    $|  output[i]=__float2bfloat16_rn(__fadd_rn(\{routed},shared[i]));
    $|}
  { text, symbol, plan, launch: launch((cells + 255L) / 256L, 256), }
}
