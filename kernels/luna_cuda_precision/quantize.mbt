///|
pub(all) enum QuantizationStage {
  GlobalScale
  BlockScales
  Pack
} derive(Debug, Eq)

///|
pub struct QuantizationSource {
  priv text : String
  priv stages : ReadOnlyArray[QuantizationStage]
  priv symbol : String
  priv representation : @format.Representation
  priv plan : @precision.QuantizationPlan
  priv launches : ReadOnlyArray[ConversionLaunch]
}

///|
pub fn QuantizationSource::source(self : QuantizationSource) -> String {
  self.text
}

///|
pub fn QuantizationSource::stages(
  self : QuantizationSource,
) -> ReadOnlyArray[QuantizationStage] {
  self.stages
}

///|
pub fn QuantizationSource::symbol(self : QuantizationSource) -> String {
  self.symbol
}

///|
pub fn QuantizationSource::representation(
  self : QuantizationSource,
) -> @format.Representation {
  self.representation
}

///|
fn encode_expression(element : @format.Element) -> String {
  match element {
    F32 => "__float_as_uint(v)"
    Bf16 =>
      "(unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(fminf(3.3895313892515355e38f,fmaxf(-3.3895313892515355e38f,v))))"
    Fp16 =>
      "(unsigned)__half_as_ushort(__float2half_rn(fminf(65504.0f,fmaxf(-65504.0f,v))))"
    E4M3 => "(unsigned)__nv_cvt_float_to_fp8(v,__NV_SATFINITE,__NV_E4M3)"
    E5M2 => "(unsigned)__nv_cvt_float_to_fp8(v,__NV_SATFINITE,__NV_E5M2)"
    Signed8 => "(unsigned)__float2int_rn(fminf(127.0f,fmaxf(-128.0f,v)))&255U"
    Signed4 => "(unsigned)__float2int_rn(fminf(7.0f,fmaxf(-8.0f,v)))&15U"
    E2M1 => "lf_precision_encode_e2(v)"
  }
}

///|
pub fn QuantizationSource::plan(
  self : QuantizationSource,
) -> @precision.QuantizationPlan {
  self.plan
}

///|
/// Dynamic activation/output/cache quantization has explicit write effects.
/// Kernels are separate ordered stages, so no inter-CTA global barrier is
/// invented. NVFP4 global amax is computed once, not redundantly per block.
pub fn quantize_source(
  plan : @precision.OperandPlan,
  symbol~ : String,
) -> QuantizationSource raise LoweringError {
  if !valid_symbol(symbol) {
    raise InvalidSymbol
  }
  if plan.placement() != QuantizeBeforeConsumer &&
    plan.placement() != QuantizeOnCacheWrite {
    raise UnsupportedPlacement
  }
  let rep = plan.representation()
  let semantic = plan.quantization() catch { _ => raise UnsupportedPlacement }
  let (rows, columns) = rep.shape()
  let (scale_rows, scale_columns) = rep.scale_shape()
  let bits = rep.element().bits()
  let (br, bc) = match rep.grouping() {
    Absent | Tensor => (rows, columns)
    Row => (1, columns)
    Block(rows=br, columns=bc) => (br, bc)
  }
  let scale_width = rep.scale_element().bytes()
  let scale_read = match rep.scale_element() {
    Absent => "1.0f"
    F32 => "__uint_as_float(lf_precision_read(scales,si*4ULL,4))"
    E4M3 => "lf_precision_e4(scales[si])"
    Ue8M0 => "lf_precision_ue8(scales[si])"
  }
  let scale_write = match rep.scale_element() {
    Absent => ""
    F32 => "lf_precision_write(scales,si*4ULL,4,__float_as_uint(requested));"
    E4M3 =>
      "const unsigned c=(unsigned)__nv_cvt_float_to_fp8(requested,__NV_SATFINITE,__NV_E4M3); scales[si]=(unsigned char)(c==0U?1U:c);"
    Ue8M0 =>
      (
        $|const unsigned u=__float_as_uint(requested), e=u>>23;
        $|const unsigned c=u<=0x00400000U?0U:e==0U?1U:e+((u&0x007fffffU)!=0U);
        $|if (c>254U) { atomicOr(error,1); return; } scales[si]=(unsigned char)c;
      )
  }
  let stages = []
  let launches = []
  let global = if semantic.passes().contains(GlobalAmax) {
    stages.push(GlobalScale)
    launches.push(launch(1L, 256))
    (
      $|extern "C" __global__ void \{symbol}_global(const float *input, float *global, int *error) {
      $|  __shared__ float maxima[256]; float a=0.0f;
      $|  for (unsigned long long i=threadIdx.x;i<\{rows.to_int64()*columns.to_int64()}ULL;i+=256ULL) { float v=input[i]; if (!isfinite(v)) atomicOr(error,1); else a=fmaxf(a,fabsf(v)); }
      $|  maxima[threadIdx.x]=a; __syncthreads();
      $|  for (int stride=128;stride>0;stride>>=1) { if (threadIdx.x<stride) maxima[threadIdx.x]=fmaxf(maxima[threadIdx.x],maxima[threadIdx.x+stride]); __syncthreads(); }
      $|  if (threadIdx.x==0) *global=maxima[0]==0.0f?1.0f:fmaxf(__fdiv_rn(maxima[0],2688.0f),__uint_as_float(1U));
      $|}
      $|
    )
  } else {
    ""
  }
  let global_value = if rep.has_global_scale() { "*global" } else { "1.0f" }
  let scales = if semantic.passes().contains(ScaleAmax) {
    stages.push(BlockScales)
    launches.push(launch(scale_rows.to_int64() * scale_columns.to_int64(), 256))
    (
      $|extern "C" __global__ void \{symbol}_scales(const float *input, unsigned char *scales, const float *global, int *error) {
      $|  const unsigned long long si=blockIdx.x; if(si>=\{scale_rows.to_int64()*scale_columns.to_int64()}ULL) return;
      $|  const unsigned long long r0=(si/\{scale_columns}ULL)*\{br}ULL, c0=(si%\{scale_columns}ULL)*\{bc}ULL;
      $|  __shared__ float maxima[256]; float a=0.0f;
      $|  for(unsigned long long i=threadIdx.x;i<\{br.to_int64()*bc.to_int64()}ULL;i+=256ULL) {
      $|    const unsigned long long r=r0+i/\{bc}ULL,c=c0+i%\{bc}ULL;
      $|    if(r<\{rows}ULL && c<\{columns}ULL) { const float v=input[r*\{columns}ULL+c]; if (!isfinite(v)) atomicOr(error,1); else a=fmaxf(a,fabsf(v)); }
      $|  }
      $|  maxima[threadIdx.x]=a; __syncthreads();
      $|  for(int stride=128;stride>0;stride>>=1) { if(threadIdx.x<stride) maxima[threadIdx.x]=fmaxf(maxima[threadIdx.x],maxima[threadIdx.x+stride]); __syncthreads(); }
      $|  if(threadIdx.x==0) { const float requested=maxima[0]==0.0f?1.0f:fmaxf(__fdiv_rn(__fdiv_rn(maxima[0],\{global_value}),__uint_as_float(\{rep.element().maximum().reinterpret_as_uint()}U)),__uint_as_float(1U)); \{scale_write} }
      $|}
      $|
    )
  } else {
    ""
  }
  stages.push(Pack)
  let units = if bits == 4 { rep.row_bytes() } else { columns.to_int64() }
  launches.push(launch((rows.to_int64() * units + 255L) / 256L, 256))
  let pair = if bits == 4 {
    (
      $|const unsigned low=\{symbol}_encode_at(input,scales,global,row,unit*2ULL,error);
      $|const unsigned high=unit*2ULL+1ULL<\{columns}ULL?\{symbol}_encode_at(input,scales,global,row,unit*2ULL+1ULL,error):0U;
      $|payload[row*\{rep.row_bytes()}ULL+unit]=(unsigned char)(low|(high<<4));
    )
  } else {
    (
      $|const unsigned code=\{symbol}_encode_at(input,scales,global,row,unit,error);
      $|lf_precision_write(payload,row*\{rep.row_bytes()}ULL+unit*\{bits/8}ULL,\{bits/8},code);
    )
  }
  let text = prelude() +
    quantization_helpers() +
    global +
    scales +
    (
      $|__device__ __forceinline__ unsigned \{symbol}_encode_at(const float *input,const unsigned char *scales,const float *global,unsigned long long row,unsigned long long column,int *error) {
      $|  const float raw=input[row*\{columns}ULL+column]; if (!isfinite(raw)) { atomicOr(error,1); return 0U; }
      $|  const unsigned long long si=\{scale_index(rep)}; const float s=\{scale_read};
      $|  const float v=\{if scale_width>0 { "__fdiv_rn(__fdiv_rn(raw,"+global_value+"),s)" } else { "raw" }};
      $|  return \{encode_expression(rep.element())};
      $|}
      $|extern "C" __global__ void \{symbol}_pack(const float *input,unsigned char *payload,const unsigned char *scales,const float *global,int *error) {
      $|  const unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
      $|  if(i>=\{rows.to_int64()*units}ULL) return; const unsigned long long row=i/\{units}ULL,unit=i%\{units}ULL;
      $|  \{pair}
      $|}
    )
  {
    text,
    symbol,
    representation: rep,
    stages: ReadOnlyArray::from_array(stages),
    plan: semantic,
    launches: ReadOnlyArray::from_array(launches),
  }
}

///|
fn quantization_helpers() -> String {
  (
    $|#ifndef LF_PRECISION_QUANTIZATION_V1
    $|#define LF_PRECISION_QUANTIZATION_V1
    $|__device__ __forceinline__ void lf_precision_write(unsigned char *p,unsigned long long i,int width,unsigned c) { for(int k=0;k<width;k++) p[i+k]=(unsigned char)(c>>(8*k)); }
    $|__device__ __forceinline__ unsigned lf_precision_encode_e2(float value) {
    $|  unsigned sign=__float_as_uint(value)>>31; float v=fminf(6.0f,fabsf(value)); unsigned low=0;
    $|  while(low<7U && lf_precision_e2(low+1U)<=v) ++low;
    $|  if(low<7U) { float a=v-lf_precision_e2(low),b=lf_precision_e2(low+1U)-v; if(a>b || (a==b && (low&1U))) ++low; }
    $|  return low|(sign<<3);
    $|}
    $|#endif
    $|
  )
}
