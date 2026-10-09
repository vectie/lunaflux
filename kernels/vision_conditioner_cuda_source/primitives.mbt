///|
pub(all) suberror SourceError {
  InvalidShape
  InvalidSymbol
} derive(Eq)

///|
fn validate(symbol : String, rows : Int, width : Int) -> Unit raise SourceError {
  guard rows > 0 &&
    width > 0 &&
    rows.to_int64() * width.to_int64() <= 2147483647L else {
    raise InvalidShape
  }
  guard !symbol.is_empty() else { raise InvalidSymbol }
  for i, c in symbol {
    guard c is ('a'..='z' | 'A'..='Z' | '_') || (i > 0 && c is ('0'..='9')) else {
      raise InvalidSymbol
    }
  }
}

///|
pub fn norm_source(
  rows~ : Int,
  width~ : Int,
  symbol~ : String,
) -> Bytes raise SourceError {
  validate(symbol, rows, width)
  @utf8.encode(
    (
      $|#include <cuda_bf16.h>
      $|#include <math.h>
      $|extern "C" __global__ void \{symbol}(const __nv_bfloat16* x,const __nv_bfloat16* weight,const __nv_bfloat16* bias,__nv_bfloat16* out) {
      $| const int row=blockIdx.x; if(row>=\{rows} || threadIdx.x) return;
      $| float mean=0.0f; for(int c=0;c<\{width};++c) mean+=__bfloat162float(x[row*\{width}+c]); mean/=\{width}.0f;
      $| float variance=0.0f; for(int c=0;c<\{width};++c) { float d=__bfloat162float(x[row*\{width}+c])-mean; variance=fmaf(d,d,variance); }
      $| const float scale=rsqrtf(variance/\{width}.0f+1e-6f);
      $| for(int c=0;c<\{width};++c) out[row*\{width}+c]=__float2bfloat16_rn((__bfloat162float(x[row*\{width}+c])-mean)*scale*__bfloat162float(weight[c])+__bfloat162float(bias[c]));
      $|}
    ),
  )
}

///|
/// Activation:0 linear,1 checkpoint tanh-GELU,2 exact merger GELU. Linear
/// output is rounded to BF16 before activation, matching separate operators.
pub fn linear_source(
  rows~ : Int,
  input_width~ : Int,
  output_width~ : Int,
  activation~ : Int,
  symbol~ : String,
) -> Bytes raise SourceError {
  validate(symbol, rows, input_width)
  validate(symbol, rows, output_width)
  guard input_width.to_int64() * output_width.to_int64() <= 2147483647L &&
    activation >= 0 &&
    activation <= 2 else {
    raise InvalidShape
  }
  let nonlinear = match activation {
    0 => ""
    1 =>
      "sum=0.5f*sum*(1.0f+tanhf(0.7978845608028654f*(sum+0.044715f*sum*sum*sum)));"
    _ => "sum=0.5f*sum*(1.0f+erff(sum*0.7071067811865475f));"
  }
  @utf8.encode(
    (
      $|#include <cuda_bf16.h>
      $|#include <math.h>
      $|extern "C" __global__ void \{symbol}(const __nv_bfloat16* x,const __nv_bfloat16* weight,const __nv_bfloat16* bias,__nv_bfloat16* out) {
      $| const int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=\{rows}*\{output_width}) return;
      $| const int row=i/\{output_width}, col=i%\{output_width}; float sum=0.0f;
      $| for(int k=0;k<\{input_width};++k) sum=fmaf(__bfloat162float(x[row*\{input_width}+k]),__bfloat162float(weight[col*\{input_width}+k]),sum);
      $| sum=__bfloat162float(__float2bfloat16_rn(sum+__bfloat162float(bias[col]))); \{nonlinear}
      $| out[i]=__float2bfloat16_rn(sum);
      $|}
    ),
  )
}

///|
pub fn residual_source(
  rows~ : Int,
  width~ : Int,
  symbol~ : String,
) -> Bytes raise SourceError {
  validate(symbol, rows, width)
  @utf8.encode(
    (
      $|#include <cuda_bf16.h>
      $|extern "C" __global__ void \{symbol}(const __nv_bfloat16* x,const __nv_bfloat16* branch,__nv_bfloat16* out) {
      $| int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<\{rows}*\{width}) out[i]=__float2bfloat16_rn(__bfloat162float(x[i])+__bfloat162float(branch[i]));
      $|}
    ),
  )
}
