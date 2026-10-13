///|
/// Terminal lowering for an ordered F32 partial, not a complete projection.
pub fn scalar_partial_source(
  rows~ : Int,
  columns~ : Int,
  inner~ : Int,
  symbol~ : String,
) -> Bytes {
  let text =
    $|#include <cuda_bf16.h>
    $|extern "C" __global__ void \{symbol}(const int *counts,const __nv_bfloat16 *input,const __nv_bfloat16 *weight,float *output) {
    $|  const unsigned long long index=(unsigned long long)blockIdx.x*256+threadIdx.x;
    $|  const int live=counts[3]; if(live<=0||live>\{rows}||index>=(unsigned long long)live*\{columns}) return;
    $|  const int row=index/\{columns},column=index%\{columns}; float sum=0.0f;
    $|  for(int k=0;k<\{inner};++k) sum=__fadd_rn(sum,__fmul_rn(__bfloat162float(input[(unsigned long long)row*\{inner}+k]),__bfloat162float(weight[(unsigned long long)column*\{inner}+k])));
    $|  output[index]=sum;
    $|}
  @utf8.encode(text)
}

///|
/// Must execute after the caller-owned reduction; no collective is hidden here.
pub fn publication_source(rows~ : Int, width~ : Int, symbol~ : String) -> Bytes {
  let text =
    $|#include <cuda_bf16.h>
    $|extern "C" __global__ void \{symbol}(const int *counts,const float *partial,__nv_bfloat16 *output) {
    $|  const unsigned long long i=(unsigned long long)blockIdx.x*256+threadIdx.x;
    $|  const int live=counts[3]; if(live<=0||live>\{rows}||i>=(unsigned long long)live*\{width}) return;
    $|  output[i]=__float2bfloat16_rn(partial[i]);
    $|}
  @utf8.encode(text)
}
