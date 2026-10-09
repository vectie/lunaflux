///|
/// BF16 attention/linear intermediates with F32 residuals and F32 scale.
/// Noncausal attention uses online F32 softmax, avoiding quadratic storage.
pub fn transformer_tail_source(
  rows~ : Int,
  hidden~ : Int,
  heads~ : Int,
  head_dimension~ : Int,
  intermediate~ : Int,
  prefix~ : String,
) -> Bytes raise EmbeddingError {
  guard heads > 0 &&
    head_dimension > 0 &&
    head_dimension <= 256 &&
    heads.to_int64() * head_dimension.to_int64() == hidden.to_int64() &&
    rows > 5 &&
    intermediate > 0 &&
    rows.to_int64() * intermediate.to_int64() <= 2147483647L else {
    raise InvalidGeometry
  }
  let checked = render(
    batch=1,
    positions=rows - 5,
    input_channels=hidden,
    hidden_channels=hidden,
    register_rows=4,
    symbol=prefix + "_attention",
  )
  let elements = checked.output_elements
  let scale = 1.0 / head_dimension.to_double().sqrt()
  @utf8.encode(
    (
      $|#include <cuda_bf16.h>
      $|#include <math.h>
      $|extern "C" __global__ void \{prefix}_attention(const __nv_bfloat16 *qkv, __nv_bfloat16 *output) {
      $| const unsigned long long group=blockIdx.x; if(group>=\{rows.to_int64()*heads.to_int64()}ULL || threadIdx.x!=0)return;
      $| const unsigned long long qr=group/\{heads}ULL, head=group%\{heads}ULL, qb=qr*\{hidden}ULL+head*\{head_dimension}ULL;
      $| float maximum=-INFINITY, denominator=0.0f, accumulator[\{head_dimension}];
      $| for(int c=0;c<\{head_dimension};++c)accumulator[c]=0.0f;
      $| for(int kr=0;kr<\{rows};++kr){
      $|  const unsigned long long kb=(unsigned long long)kr*\{hidden}ULL+head*\{head_dimension}ULL;float dot=0.0f;
      $|  for(int c=0;c<\{head_dimension};++c)dot=__fadd_rn(dot,__fmul_rn(__bfloat162float(qkv[qb+c]),__bfloat162float(qkv[\{elements}ULL+kb+c])));
      $|  const float score=__fmul_rn(dot,\{scale}f), next=fmaxf(maximum,score), alpha=expf(__fsub_rn(maximum,next)), beta=expf(__fsub_rn(score,next));
      $|  denominator=__fadd_rn(__fmul_rn(denominator,alpha),beta);
      $|  for(int c=0;c<\{head_dimension};++c)accumulator[c]=__fadd_rn(__fmul_rn(accumulator[c],alpha),__fmul_rn(beta,__bfloat162float(qkv[\{elements*2L}ULL+kb+c])));
      $|  maximum=next;
      $| }
      $| for(int c=0;c<\{head_dimension};++c)output[qb+c]=__float2bfloat16_rn(__fdiv_rn(accumulator[c],denominator));
      $|}
      $|extern "C" __global__ void \{prefix}_project(const __nv_bfloat16 *input,const float *weight,const float *bias,__nv_bfloat16 *output){
      $| const unsigned long long i=(unsigned long long)blockIdx.x*256ULL+threadIdx.x;if(i>=\{elements}ULL)return;const unsigned long long row=i/\{hidden}ULL,c=i%\{hidden}ULL;float sum=0.0f;
      $| for(int k=0;k<\{hidden};++k)sum=__fadd_rn(sum,__fmul_rn(__bfloat162float(input[row*\{hidden}ULL+k]),__bfloat162float(__float2bfloat16_rn(weight[c*\{hidden}ULL+k]))));
      $| output[i]=__float2bfloat16_rn(__fadd_rn(sum,__bfloat162float(__float2bfloat16_rn(bias[c]))));
      $|}
      $|extern "C" __global__ void \{prefix}_residual(const float *input,const __nv_bfloat16 *delta,const float *scale,float *output){
      $| const unsigned long long i=(unsigned long long)blockIdx.x*256ULL+threadIdx.x;if(i>=\{elements}ULL)return;
      $| output[i]=__fadd_rn(input[i],__fmul_rn(__bfloat162float(delta[i]),scale[i%\{hidden}ULL]));
      $|}
      $|extern "C" __global__ void \{prefix}_gate_up(const __nv_bfloat16 *input,const float *weight,const float *bias,__nv_bfloat16 *output){
      $| const unsigned long long i=(unsigned long long)blockIdx.x*256ULL+threadIdx.x;if(i>=\{rows.to_int64()*intermediate.to_int64()}ULL)return;const unsigned long long row=i/\{intermediate}ULL,c=i%\{intermediate}ULL;float gate=0.0f,up=0.0f;
      $| for(int k=0;k<\{hidden};++k){const float x=__bfloat162float(input[row*\{hidden}ULL+k]);gate=__fadd_rn(gate,__fmul_rn(x,__bfloat162float(__float2bfloat16_rn(weight[c*\{hidden}ULL+k]))));up=__fadd_rn(up,__fmul_rn(x,__bfloat162float(__float2bfloat16_rn(weight[(c+\{intermediate}ULL)*\{hidden}ULL+k]))));}
      $| gate=__bfloat162float(__float2bfloat16_rn(__fadd_rn(gate,__bfloat162float(__float2bfloat16_rn(bias[c])))));up=__bfloat162float(__float2bfloat16_rn(__fadd_rn(up,__bfloat162float(__float2bfloat16_rn(bias[c+\{intermediate}ULL])))));
      $| const float activated=__bfloat162float(__float2bfloat16_rn(__fdiv_rn(gate,__fadd_rn(1.0f,expf(-gate)))));output[i]=__float2bfloat16_rn(__fmul_rn(activated,up));
      $|}
      $|extern "C" __global__ void \{prefix}_down(const __nv_bfloat16 *input,const float *weight,const float *bias,__nv_bfloat16 *output){
      $| const unsigned long long i=(unsigned long long)blockIdx.x*256ULL+threadIdx.x;if(i>=\{elements}ULL)return;const unsigned long long row=i/\{hidden}ULL,c=i%\{hidden}ULL;float sum=0.0f;
      $| for(int k=0;k<\{intermediate};++k)sum=__fadd_rn(sum,__fmul_rn(__bfloat162float(input[row*\{intermediate}ULL+k]),__bfloat162float(__float2bfloat16_rn(weight[c*\{intermediate}ULL+k]))));
      $| output[i]=__float2bfloat16_rn(__fadd_rn(sum,__bfloat162float(__float2bfloat16_rn(bias[c]))));
      $|}
    ),
  )
}
