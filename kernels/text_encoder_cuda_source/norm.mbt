///|
/// Qwen RMSNorm: F32 reduction, BF16 normalized intermediate, BF16 multiply.
pub fn norm_source(
  rows~ : Int,
  width~ : Int,
  epsilon~ : Double,
  symbol~ : String,
) -> Bytes raise TextEncoderSourceError {
  guard rows > 0 && rows <= 262144 && width > 0 && width <= 65536 else {
    raise InvalidShape
  }
  guard symbol_ok(symbol) else { raise InvalidSymbol }
  guard epsilon > 0.0 && epsilon <= 1.0 && !epsilon.is_nan() else {
    raise InvalidEpsilon
  }
  let eps = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(epsilon)
  @utf8.encode(
    (
      $|#include <cuda_bf16.h>
      $|#include <math.h>
      $|extern "C" __global__ void \{symbol}(const __nv_bfloat16 *input, const __nv_bfloat16 *weight, __nv_bfloat16 *output) {
      $|  const int row=blockIdx.x;
      $|  if(row>=\{rows} || threadIdx.x!=0) return;
      $|  const unsigned long long base=(unsigned long long)row*\{width};
      $|  float sum=0.0f;
      $|  for(int d=0;d<\{width};++d) { float v=__bfloat162float(input[base+d]); sum=__fadd_rn(sum,__fmul_rn(v,v)); }
      $|  const float inv=rsqrtf(__fadd_rn(__fdiv_rn(sum,\{width}.0f),\{eps}));
      $|  for(int d=0;d<\{width};++d) {
      $|    const float normalized=__bfloat162float(__float2bfloat16_rn(__fmul_rn(__bfloat162float(input[base+d]),inv)));
      $|    output[base+d]=__float2bfloat16_rn(__fmul_rn(normalized,__bfloat162float(weight[d])));
      $|  }
      $|}
    ),
  )
}
