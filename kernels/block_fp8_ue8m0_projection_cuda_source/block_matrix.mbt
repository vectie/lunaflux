///|
/// Fixed-size scratch keeps the arithmetic path legal for every supported
/// input width. One raw block is encoded once per CTA, then all output warps
/// consume it before the next block replaces it.
fn matrix_projection_source(
  input_width~ : Int,
  output_width~ : Int,
  maximum_rows~ : Int,
  function_symbol~ : String,
  scaled_dot_law~ : @precision.ScaledDotReductionLaw,
) -> Bytes raise BlockFp8Ue8M0ProjectionCudaSourceError {
  guard scaled_dot_law
    is UnscaledBlockMatrixFloat32(
      block_columns=128,
      publication=SeparateMultiplyAdd
    ) else {
    raise UnsupportedArithmetic
  }
  let helper = @lowering.e4m3_block_matrix_helper(
    symbol=function_symbol,
    publication=SeparateMultiplyAdd,
  )
  @utf8.encode(
    helper +
    (
      $|
      $|extern "C" __global__ void \{function_symbol}(const int *counts,const __nv_bfloat16 *input,const unsigned char *weight,const unsigned char *weight_scale,__nv_bfloat16 *output) {
      $|  const int live=counts[3],row=(int)blockIdx.x;
      $|  if(live<=0||live>\{maximum_rows}||row>=live)return;
      $|  const int lane=threadIdx.x&31,warp=threadIdx.x/32,g=lane/4,t=lane%4;
      $|  const int tile=(int)blockIdx.y*32+warp*8;
      $|  __shared__ unsigned char encoded[128],encoded_scale[1];
      $|  __shared__ float maxima[4];
      $|  float total[2]={0.0f,0.0f};
      $|  for(int group=0;group<\{input_width/128};++group) {
      $|    const unsigned long long base=(unsigned long long)row*\{input_width}ULL+group*128;
      $|    const float value=__bfloat162float(input[base+threadIdx.x]);
      $|    float maximum=fmaxf(0.0001f,fabsf(value));
      $|    for(int delta=16;delta>0;delta>>=1)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffU,maximum,delta));
      $|    if(lane==0)maxima[warp]=maximum;
      $|    __syncthreads();
      $|    maximum=fmaxf(fmaxf(maxima[0],maxima[1]),fmaxf(maxima[2],maxima[3]));
      $|    // Close every reader's scratch lifetime before quantization publication.
      $|    // CUDA may physically alias dead maxima with the encoded scale.
      $|    __syncthreads();
      $|    const unsigned bits=__float_as_uint(__fmul_rn(maximum,0.002232142857142857f));
      $|    const unsigned scale_bits=((bits>>23)&255U)+((bits&0x7fffffU)!=0U?1U:0U);
      $|    const float scale=__uint_as_float(scale_bits<<23);
      $|    const float divided=__fdiv_rn(value,scale);
      $|    const float bounded=fminf(448.0f,fmaxf(-448.0f,divided));
      $|    encoded[threadIdx.x]=__nv_cvt_float_to_fp8(bounded==0.0f?0.0f:bounded,__NV_SATFINITE,__NV_E4M3);
      $|    if(threadIdx.x==0)encoded_scale[0]=(unsigned char)scale_bits;
      $|    __syncthreads();
      $|    float block[2];
      $|    \{function_symbol}_block_dot<128,\{input_width}>(encoded,encoded_scale,weight+group*128,weight_scale+group,tile,block);
      $|    if(g==0)for(int i=0;i<2;++i)total[i]=__fadd_rn(total[i],block[i]);
      $|    __syncthreads();
      $|  }
      $|  if(g==0)for(int i=0;i<2;++i)output[(unsigned long long)row*\{output_width}ULL+tile+t*2+i]=__float2bfloat16_rn(total[i]);
      $|}
    ),
  )
}
