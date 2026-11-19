///|
fn validate(symbol : String, rows : Int32, width : Int32) -> Unit raise {
  guard rows > 0 &&
    width > 0 &&
    rows.to_int64() * width.to_int64() <= 2147483647L else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  guard !symbol.is_empty() &&
    symbol.length() <= 128 &&
    symbol[0] is ('a'..<='z' | 'A'..<='Z' | '_') &&
    symbol.all(c => c is ('a'..<='z' | 'A'..<='Z' | '0'..<='9' | '_')) else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
}

///|
/// NaViT [1,C,P,live_rows*P] -> row-major [live_rows,C*P*P].
pub fn patch_rows_source(
  maximum_rows~ : Int32,
  channels~ : Int32,
  patch_size~ : Int32,
  symbol~ : String,
) -> Bytes raise {
  guard channels > 0 &&
    patch_size > 0 &&
    channels.to_int64() * patch_size.to_int64() * patch_size.to_int64() <=
    2147483647L else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let width = channels * patch_size * patch_size
  validate(symbol, maximum_rows, width)
  @utf8.encode(
    (
      $|#include <cuda_bf16.h>
      $|#include <stdint.h>
      $|extern "C" __global__ void \{symbol}(const int32_t* counts,const __nv_bfloat16* pixels,__nv_bfloat16* out){
      $| const int live=counts[3],i=blockIdx.x*blockDim.x+threadIdx.x;
      $| if(live<=0||live>\{maximum_rows}||i>=live*\{width})return;
      $| const int row=i/\{width},inner=i%\{width},channel=inner/\{patch_size*patch_size},py=inner%\{patch_size*patch_size}/\{patch_size},px=inner%\{patch_size};
      $| reinterpret_cast<uint16_t*>(out)[i]=reinterpret_cast<const uint16_t*>(pixels)[(channel*\{patch_size}+py)*(unsigned long long)live*\{patch_size}+row*\{patch_size}+px];
      $|}
    ),
  )
}

///|
/// One learned absolute position lookup/add per row. IDs are admitted by the
/// layout owner; there is no interpolation or vision rotary transformation.
pub fn position_add_source(
  maximum_rows~ : Int32,
  width~ : Int32,
  symbol~ : String,
) -> Bytes raise {
  validate(symbol, maximum_rows, width)
  @utf8.encode(
    (
      $|#include <cuda_bf16.h>
      $|#include <stdint.h>
      $|extern "C" __global__ void \{symbol}(const int32_t* counts,const int32_t* ids,const __nv_bfloat16* positions,__nv_bfloat16* state){
      $| const int live=counts[3],i=blockIdx.x*blockDim.x+threadIdx.x;
      $| if(live<=0||live>\{maximum_rows}||i>=live*\{width})return;
      $| state[i]=__float2bfloat16_rn(__fadd_rn(__bfloat162float(state[i]),__bfloat162float(positions[(unsigned long long)ids[i/\{width}]*\{width}+i%\{width}])));
      $|}
    ),
  )
}

///|
/// Raw BF16 row permutation, shared by early attention windows and mergers.
pub fn gather_source(
  maximum_rows~ : Int32,
  width~ : Int32,
  symbol~ : String,
) -> Bytes raise {
  validate(symbol, maximum_rows, width)
  @utf8.encode(
    (
      $|#include <cuda_bf16.h>
      $|#include <stdint.h>
      $|extern "C" __global__ void \{symbol}(const int32_t* counts,const int32_t* indices,const __nv_bfloat16* input,__nv_bfloat16* out){
      $| const int live=counts[3],i=blockIdx.x*blockDim.x+threadIdx.x;
      $| if(live<=0||live>\{maximum_rows}||i>=live*\{width})return;
      $| reinterpret_cast<uint16_t*>(out)[i]=reinterpret_cast<const uint16_t*>(input)[(unsigned long long)indices[i/\{width}]*\{width}+i%\{width}];
      $|}
    ),
  )
}

///|
/// Add mean of each four input rows to one merger row, with one BF16 publish.
pub fn merge_residual_source(
  maximum_rows~ : Int32,
  width~ : Int32,
  symbol~ : String,
) -> Bytes raise {
  validate(symbol, maximum_rows, width)
  guard maximum_rows % 4 == 0 else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  @utf8.encode(
    (
      $|#include <cuda_bf16.h>
      $|#include <stdint.h>
      $|extern "C" __global__ void \{symbol}(const int32_t* counts,const __nv_bfloat16* windows,const __nv_bfloat16* branch,__nv_bfloat16* out){
      $| const int live=counts[3],i=blockIdx.x*blockDim.x+threadIdx.x;
      $| if(live<=0||live>\{maximum_rows}||live%4||i>=live/4*\{width})return;
      $| const int group=i/\{width},c=i%\{width};float sum=0.0f;
      $| for(int p=0;p<4;++p)sum=__fadd_rn(sum,__bfloat162float(windows[(group*4+p)*\{width}+c]));
      $| const float residual=__bfloat162float(__float2bfloat16_rn(__fmul_rn(sum,0.25f)));
      $| out[i]=__float2bfloat16_rn(__fadd_rn(__bfloat162float(branch[i]),residual));
      $|}
    ),
  )
}
