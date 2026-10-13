///|
/// Two ordered kernels implement the IR's global-amax then block-amax/pack
/// effects. The returned source uses symbol_global and symbol_pack. No host
/// readback or cross-CTA barrier is used, and only live rows enter the amax.
pub fn quantize_fp4_bf16_rows_source(
  plan : @precision.QuantizationPlan,
  row_multiplier~ : Int,
  symbol~ : String,
) -> String raise LoweringError {
  let operand = plan.operand()
  let rep = operand.representation()
  let (rows, columns) = rep.shape()
  guard valid_symbol(symbol) &&
    operand.role() == Activation &&
    operand.placement() == QuantizeBeforeConsumer &&
    rep.format() == NvFp4 &&
    columns % 64 == 0 &&
    row_multiplier > 0 &&
    rows % row_multiplier == 0 else {
    raise UnsupportedPlacement
  }
  let storage = @precision.packed_buffer_layout(rep) catch {
    _ => raise UnsupportedPlacement
  }
  let scale = storage.scales_offset().unwrap()
  let global = storage.global_scale_offset().unwrap()
  let floor = Float::from_double(plan.amax_floor()).reinterpret_as_uint()
  prelude() +
  quantization_helpers() +
  (
    $|extern "C" __global__ void \{symbol}_global(const int *counts,const __nv_bfloat16 *x,unsigned char *packed,int *error) {
    $|  int rows=counts[3]; if(rows<=0||rows>\{rows/row_multiplier}) return;
    $|  __shared__ float maxima[256]; float m=__uint_as_float(\{floor}U);
    $|  for(unsigned long long i=threadIdx.x;i<(unsigned long long)rows*\{row_multiplier}*\{columns};i+=256) {float v=__bfloat162float(x[i]);if(!isfinite(v)) atomicOr(error,1);else m=fmaxf(m,fabsf(v));}
    $|  maxima[threadIdx.x]=m; __syncthreads();
    $|  for(int d=128;d;d>>=1) {if(threadIdx.x<d) maxima[threadIdx.x]=fmaxf(maxima[threadIdx.x],maxima[threadIdx.x+d]);__syncthreads();}
    $|  if(threadIdx.x==0) *(float *)(packed+\{global}ULL)=maxima[0]==0.0f?1.0f:fmaxf(__fdiv_rn(maxima[0],2688.0f),__uint_as_float(1U));
    $|}
    $|extern "C" __global__ void \{symbol}_pack(const int *counts,const __nv_bfloat16 *x,unsigned char *packed,int *error) {
    $|  int rows=counts[3]; if(rows<=0||rows>\{rows/row_multiplier}) return;
    $|  unsigned long long cell=(unsigned long long)blockIdx.x*128+threadIdx.x;
    $|  if(cell>=(unsigned long long)rows*\{row_multiplier}*\{columns}) return;
    $|  float raw=__bfloat162float(x[cell]); if(!isfinite(raw)) {atomicOr(error,1);raw=0.0f;}
    $|  float m=fmaxf(fabsf(raw),__uint_as_float(\{floor}U));
    $|  for(int d=8;d;d>>=1) m=fmaxf(m,__shfl_xor_sync(0xffffffffU,m,d,16));
    $|  float global=*(const float *)(packed+\{global}ULL);
    $|  float requested=m==0.0f?1.0f:fmaxf(__fdiv_rn(__fdiv_rn(m,global),6.0f),__uint_as_float(1U));
    $|  unsigned code=(unsigned)__nv_cvt_float_to_fp8(requested,__NV_SATFINITE,__NV_E4M3); if(!code) code=1;
    $|  if((cell&15ULL)==0) packed[\{scale}ULL+cell/16]=(unsigned char)code;
    $|  float v=__fdiv_rn(__fdiv_rn(raw,global),lf_precision_e4(code));
    $|  unsigned q=lf_precision_encode_e2(v),other=__shfl_xor_sync(0xffffffffU,q,1);
    $|  if((cell&1ULL)==0) packed[cell/2]=(unsigned char)(q|(other<<4));
    $|}
  )
}
