///|
/// Two ordered kernels implement the IR's global-amax then block-amax/pack
/// effects. The returned source uses symbol_global and symbol_pack. No host
/// readback or cross-CTA barrier is used, and only live rows enter the amax.
pub fn quantize_fp4_bf16_rows_source(
  plan : @precision.QuantizationPlan,
  row_multiplier~ : Int,
  symbol~ : String,
  maximum? : @physical.PartitionedMaximum,
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
  let tree = @physical.MaximumReductionTree::new(leaves=256, group=32).unwrap()
  let (leaves, group, groups) = tree.geometry()
  if maximum is Some(p) {
    guard p.group_elements() == leaves &&
      p.partial_offset() >= storage.buffer_bytes() else {
      raise UnsupportedPlacement
    }
  }
  let scan_setup = match maximum {
    None => ""
    Some(p) =>
      "int producers=rows==1?1:\{p.partitions()}; if(blockIdx.x>=producers) return;"
  }
  let start = if maximum is Some(_) {
    "blockIdx.x*\{leaves}+threadIdx.x"
  } else {
    "threadIdx.x"
  }
  let stride = if maximum is Some(_) {
    "producers*\{leaves}"
  } else {
    "\{leaves}"
  }
  let publication = match maximum {
    None =>
      "*(float *)(packed+\{global}ULL)=m==0.0f?1.0f:fmaxf(__fdiv_rn(m,2688.0f),__uint_as_float(1U));"
    Some(p) => "((float *)(packed+\{p.partial_offset()}ULL))[blockIdx.x]=m;"
  }
  let read_global = match maximum {
    None => "float global=*(const float *)(packed+\{global}ULL);"
    Some(p) =>
      (
        $|float total=__uint_as_float(\{floor}U); int producers=rows==1?1:\{p.partitions()};
        $|for(int i=threadIdx.x&31;i<producers;i+=32) total=fmaxf(total,((const float *)(packed+\{p.partial_offset()}ULL))[i]);
        $|for(int d=16;d;d>>=1) total=fmaxf(total,__shfl_down_sync(0xffffffffU,total,d));
        $|total=__shfl_sync(0xffffffffU,total,0);
        $|float global=total==0.0f?1.0f:fmaxf(__fdiv_rn(total,2688.0f),__uint_as_float(1U));
        $|if(cell==0) *(float *)(packed+\{global}ULL)=global;
      )
  }
  prelude() +
  quantization_helpers() +
  (
    $|extern "C" __global__ void \{symbol}_global(const int *counts,const __nv_bfloat16 *x,unsigned char *packed,int *error) {
    $|  int rows=counts[3]; if(rows<=0||rows>\{rows/row_multiplier}) return;
    $|  \{scan_setup}
    $|  __shared__ float maxima[\{groups}]; float m=__uint_as_float(\{floor}U);
    $|  for(unsigned long long i=\{start};i<(unsigned long long)rows*\{row_multiplier}*\{columns};i+=\{stride}) {float v=__bfloat162float(x[i]);if(!isfinite(v)) atomicOr(error,1);else m=fmaxf(m,fabsf(v));}
    $|  for(int d=\{group/2};d;d>>=1) m=fmaxf(m,__shfl_down_sync(0xffffffffU,m,d));
    $|  if((threadIdx.x%\{group})==0) maxima[threadIdx.x/\{group}]=m;
    $|  __syncthreads();
    $|  if(threadIdx.x<\{group}) {
    $|    m=threadIdx.x<\{groups}?maxima[threadIdx.x]:__uint_as_float(\{floor}U);
    $|    for(int d=\{group/2};d;d>>=1) m=fmaxf(m,__shfl_down_sync(0xffffffffU,m,d));
    $|    if(threadIdx.x==0) \{publication}
    $|  }
    $|}
    $|extern "C" __global__ void \{symbol}_pack(const int *counts,const __nv_bfloat16 *x,unsigned char *packed,int *error) {
    $|  int rows=counts[3]; if(rows<=0||rows>\{rows/row_multiplier}) return;
    $|  unsigned long long cell=(unsigned long long)blockIdx.x*128+threadIdx.x;
    $|  if(cell>=(unsigned long long)rows*\{row_multiplier}*\{columns}) return;
    $|  float raw=__bfloat162float(x[cell]); if(!isfinite(raw)) {atomicOr(error,1);raw=0.0f;}
    $|  float m=fmaxf(fabsf(raw),__uint_as_float(\{floor}U));
    $|  for(int d=8;d;d>>=1) m=fmaxf(m,__shfl_xor_sync(0xffffffffU,m,d,16));
    $|  \{read_global}
    $|  float requested=m==0.0f?1.0f:fmaxf(__fdiv_rn(__fdiv_rn(m,global),6.0f),__uint_as_float(1U));
    $|  unsigned code=(unsigned)__nv_cvt_float_to_fp8(requested,__NV_SATFINITE,__NV_E4M3); if(!code) code=1;
    $|  if((cell&15ULL)==0) packed[\{scale}ULL+cell/16]=(unsigned char)code;
    $|  float v=__fdiv_rn(__fdiv_rn(raw,global),lf_precision_e4(code));
    $|  unsigned q=lf_precision_encode_e2(v),other=__shfl_xor_sync(0xffffffffU,q,1);
    $|  if((cell&1ULL)==0) packed[cell/2]=(unsigned char)(q|(other<<4));
    $|}
  )
}
