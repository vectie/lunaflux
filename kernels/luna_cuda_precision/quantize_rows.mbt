///|
pub struct RowQuantizationSource {
  priv text : String
  priv symbol : String
  priv launch : ConversionLaunch
  priv plan : @precision.QuantizationPlan
}

///|
pub fn RowQuantizationSource::source(self : Self) -> String {
  self.text
}

///|
pub fn RowQuantizationSource::symbol(self : Self) -> String {
  self.symbol
}

///|
pub fn RowQuantizationSource::launch(self : Self) -> ConversionLaunch {
  self.launch
}

///|
pub fn RowQuantizationSource::plan(self : Self) -> @precision.QuantizationPlan {
  self.plan
}

///|
/// Quantize BF16 rows into one compact payload/scale allocation. Each CTA owns
/// a complete scale block, so scale publication and payload encoding need no
/// inter-CTA synchronization. Counts[3] supplies live original rows; multiplier
/// accounts for selected-expert products without a host diagnostic round trip.
/// The error word is a caller-owned device status, cleared before execution.
pub fn quantize_bf16_rows_source(
  plan : @precision.QuantizationPlan,
  row_multiplier~ : Int,
  symbol~ : String,
) -> RowQuantizationSource raise LoweringError {
  if !valid_symbol(symbol) {
    raise InvalidSymbol
  }
  let operand = plan.operand()
  let rep = operand.representation()
  let (rows, columns) = rep.shape()
  guard operand.role() == Activation &&
    operand.placement() == QuantizeBeforeConsumer &&
    rep.scale_element() == Ue8M0 &&
    (rep.element() == E4M3 || rep.element() == E5M2) &&
    rep.grouping() is Block(rows=1, columns=block_columns) &&
    row_multiplier > 0 &&
    rows % row_multiplier == 0 else {
    raise UnsupportedPlacement
  }
  let (_, scale_columns) = rep.scale_shape()
  let layout = @precision.packed_buffer_layout(rep) catch {
    _ => raise UnsupportedPlacement
  }
  let scale_offset = layout.scales_offset().unwrap()
  let floor_bits = Float::from_double(plan.amax_floor()).reinterpret_as_uint()
  let reciprocal_bits = (1.0F / rep.element().maximum()).reinterpret_as_uint()
  let text = prelude() +
    (
      $|extern "C" __global__ void \{symbol}(const int *counts, const __nv_bfloat16 *input, unsigned char *packed, int *error) {
      $|  const int live_rows=counts[3]; if(live_rows<=0||live_rows>\{rows/row_multiplier}) return;
      $|  const unsigned long long block=blockIdx.x, row=block/\{scale_columns}ULL;
      $|  if(row>=(unsigned long long)live_rows*\{row_multiplier}ULL) return;
      $|  const unsigned long long c0=(block%\{scale_columns}ULL)*\{block_columns}ULL;
      $|  __shared__ float maxima[128]; __shared__ float scale;
      $|  float a=__uint_as_float(\{floor_bits}U);
      $|  for(unsigned long long c=threadIdx.x;c<\{block_columns}ULL;c+=128ULL) {
      $|    if(c0+c<\{columns}ULL) { const float v=__bfloat162float(input[row*\{columns}ULL+c0+c]); if(!isfinite(v)) atomicOr(error,1); else a=fmaxf(a,fabsf(v)); }
      $|  }
      $|  maxima[threadIdx.x]=a; __syncthreads();
      $|  for(int stride=64;stride>0;stride>>=1) { if(threadIdx.x<stride) maxima[threadIdx.x]=fmaxf(maxima[threadIdx.x],maxima[threadIdx.x+stride]); __syncthreads(); }
      $|  if(threadIdx.x==0) {
      $|    const float requested=maxima[0]==0.0f?1.0f:__fmul_rn(maxima[0],__uint_as_float(\{reciprocal_bits}U));
      $|    const unsigned u=__float_as_uint(requested), e=u>>23;
      $|    const unsigned code=u<=0x00400000U?0U:e==0U?1U:e+((u&0x007fffffU)!=0U);
      $|    if(code>254U) { atomicOr(error,1); scale=1.0f; packed[\{scale_offset}ULL+block]=127U; }
      $|    else { packed[\{scale_offset}ULL+block]=(unsigned char)code; scale=lf_precision_ue8(code); }
      $|  }
      $|  __syncthreads();
      $|  for(unsigned long long c=threadIdx.x;c<\{block_columns}ULL;c+=128ULL) {
      $|    if(c0+c<\{columns}ULL) {
      $|      const float raw=__bfloat162float(input[row*\{columns}ULL+c0+c]);
      $|      const float v=isfinite(raw)?__fdiv_rn(raw,scale):0.0f;
      $|      packed[row*\{rep.row_bytes()}ULL+c0+c]=(unsigned char)(\{encode_expression(rep.element())});
      $|    }
      $|  }
      $|}
    )
  {
    text,
    symbol,
    launch: launch(rows.to_int64() * scale_columns.to_int64(), 128),
    plan,
  }
}
