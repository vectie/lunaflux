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
  let (_, _, _, rows, _) = plan.rotary().geometry()
  let (groups, _, rank, _) = plan.geometry()
  let symbols = [
    symbol_prefix + "_inverse",
    symbol_prefix + "_grouped",
    symbol_prefix + "_output",
  ]
  let inverse = @rotary.rotary_tensor_source(plan.rotary(), symbol=symbols[0])
  let output_b = @projection.block_fp8_ue8m0_projection_cuda_source(
    input_width=groups * rank,
    output_width=plan.hidden_width(),
    maximum_rows=rows,
    function_symbol=symbols[2],
  )
  let source = output_b.source_bytes() +
    @utf8.encode(inverse) +
    @utf8.encode(grouped_dot_source(plan, symbols[1]))
  { plan, source, symbols: ReadOnlyArray::from_array(symbols), }
}

///|
fn grouped_dot_source(
  plan : @precision.GroupedAttentionOutputPrecision,
  symbol : String,
) -> String raise {
  let (groups, input, rank, rows) = plan.geometry()
  match plan.accumulation() {
    CompensatedFloat32 | GroupedVectorFloat32(..) =>
      raise @precision.PrecisionPlanError::NoImplementation
    StripedTreeFloat32(partitions~) =>
      striped_dot_source(plan, symbol, partitions)
    OrderedFloat32 =>
      (
        $|extern "C" __global__ void \{symbol}(const int32_t *counts, const __nv_bfloat16 *input, const unsigned char *weight, const unsigned char *scales, __nv_bfloat16 *output) {
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
      )
  }
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
    1 =>
      match self.plan.accumulation() {
        CompensatedFloat32 | GroupedVectorFloat32(..) => (rows, groups, 256) // Unreachable: source factory rejects this law.
        OrderedFloat32 => (rows, groups, 256)
        StripedTreeFloat32(partitions~) => {
          let (_, _, rank, _) = self.plan.geometry()
          (
            rows,
            groups * ((rank + 256 / partitions - 1) / (256 / partitions)),
            256,
          )
        }
      }
    _ => (rows, (self.plan.hidden_width() + 255) / 256, 256)
  }
}
