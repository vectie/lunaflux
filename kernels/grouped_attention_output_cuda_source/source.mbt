///|
pub struct GroupedAttentionOutputCudaSource {
  priv plan : @precision.GroupedAttentionOutputPrecision
  priv source : Bytes
  priv symbols : ReadOnlyArray[String]
}

///|
pub fn grouped_attention_output_source(
  plan : @precision.GroupedAttentionOutputPrecision,
  symbol_prefix~ : String,
) -> GroupedAttentionOutputCudaSource raise {
  guard !symbol_prefix.is_empty() &&
    symbol_prefix.length() <= 112 &&
    symbol_prefix[0] is ('a'..='z' | 'A'..='Z' | '_') &&
    symbol_prefix.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_')) else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let (heads, width, suffix, rows, positions) = plan.rotary().geometry()
  let (groups, input, rank, _) = plan.geometry()
  let symbols = [
    symbol_prefix + "_inverse",
    symbol_prefix + "_grouped",
    symbol_prefix + "_output",
  ]
  let frequency = @rotary.rotary_frequency_source(
    plan.rotary().frequency(),
    suffix,
  )
  let output_b = @projection.block_fp8_ue8m0_projection_cuda_source(
    input_width=groups * rank,
    output_width=plan.hidden_width(),
    maximum_rows=rows,
    function_symbol=symbols[2],
  )
  let source = output_b.source_bytes() +
    @utf8.encode(
      (
        $|extern "C" __global__ void \{symbols[0]}(const int32_t *counts, const int32_t *positions, const uint16_t *input_words, uint16_t *output_words) {
        $|  const int row = blockIdx.x, head = blockIdx.y, live = counts[3];
        $|  if (live <= 0 || live > \{rows} || row >= live || head >= \{heads}) return;
        $|  const int position = positions[row]; if (position < 0 || position >= \{positions}) return;
        $|  const unsigned long long base = ((unsigned long long)row * \{heads}ULL + head) * \{width}ULL;
        $|  for (int col = threadIdx.x; col < \{width - suffix}; col += blockDim.x) output_words[base + col] = input_words[base + col];
        $|  for (int pair = threadIdx.x; pair < \{suffix / 2}; pair += blockDim.x) {
        $|    \{frequency}
        $|    const float angle = __fmul_rn((float)position, frequency), cosine = cosf(angle), sine = -sinf(angle);
        $|    const unsigned long long index = base + \{width - suffix}ULL + pair * 2ULL;
        $|    const __nv_bfloat16 *input = (const __nv_bfloat16 *)input_words; __nv_bfloat16 *output = (__nv_bfloat16 *)output_words;
        $|    const float left = __bfloat162float(input[index]), right = __bfloat162float(input[index + 1ULL]);
        $|    output[index] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(left, cosine), -__fmul_rn(right, sine)));
        $|    output[index + 1ULL] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(left, sine), __fmul_rn(right, cosine)));
        $|  }
        $|}
        $|extern "C" __global__ void \{symbols[1]}(const int32_t *counts, const __nv_bfloat16 *input, const unsigned char *weight, const unsigned char *scales, __nv_bfloat16 *output) {
        $|  const int row = blockIdx.x, group = blockIdx.y, live = counts[3];
        $|  if (live <= 0 || live > \{rows} || row >= live || group >= \{groups}) return;
        $|  for (int column = threadIdx.x; column < \{rank}; column += blockDim.x) {
        $|    const unsigned long long input_base = ((unsigned long long)row * \{groups}ULL + group) * \{input}ULL;
        $|    const int weight_row = group * \{rank} + column; const unsigned long long weight_base = (unsigned long long)weight_row * \{input}ULL;
        $|    float sum = 0.0f;
        $|    for (int inner = 0; inner < \{input}; ++inner) {
        $|      const unsigned char bits = weight[weight_base + inner], scale = scales[(unsigned long long)(weight_row / 128) * \{input / 128}ULL + inner / 128];
        $|      if (bits == 0x7fU || bits == 0xffU || scale == 0xffU) { sum = nanf(""); break; }
        $|      const float parameter = __bfloat162float(__float2bfloat16_rn(__fmul_rn(lf_block_e4m3_decode(bits), lf_block_e8m0_decode(scale))));
        $|      sum = __fadd_rn(sum, __fmul_rn(__bfloat162float(input[input_base + inner]), parameter));
        $|    }
        $|    output[((unsigned long long)row * \{groups}ULL + group) * \{rank}ULL + column] = __float2bfloat16_rn(sum);
        $|  }
        $|}
        $|
      ),
    )
  { plan, source, symbols: ReadOnlyArray::from_array(symbols), }
}

///|
pub fn GroupedAttentionOutputCudaSource::plan(
  self : Self,
) -> @precision.GroupedAttentionOutputPrecision {
  self.plan
}

///|
pub fn GroupedAttentionOutputCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn GroupedAttentionOutputCudaSource::symbol(
  self : Self,
  stage : Int,
) -> String {
  self.symbols[stage]
}

///|
pub fn GroupedAttentionOutputCudaSource::launch(
  self : Self,
  stage : Int,
) -> (Int, Int, Int) {
  let (heads, _, _, rows, _) = self.plan.rotary().geometry()
  let (groups, _, _, _) = self.plan.geometry()
  match stage {
    0 => (rows, heads, 256)
    1 => (rows, groups, 256)
    _ => (rows, (self.plan.hidden_width() + 255) / 256, 256)
  }
}
