///|
/// CUDA subgroup lowering of the explicit, device-neutral reduction tree.
/// Every lane executes each shuffle; only the tree's surviving half consumes it.
fn striped_dot_source(
  plan : @precision.GroupedAttentionOutputPrecision,
  symbol : String,
  partitions : Int,
) -> String raise {
  guard partitions <= 32 else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let (groups, input, rank, rows) = plan.geometry()
  let columns = 256 / partitions
  let tiles = (rank + columns - 1) / columns
  (
    $|extern "C" __global__ void \{symbol}(const int32_t *counts, const __nv_bfloat16 *input, const unsigned char *weight, const unsigned char *scales, __nv_bfloat16 *output) {
    $|  const int row = blockIdx.x, group = blockIdx.y / \{tiles}, tile = blockIdx.y % \{tiles}, live = counts[3];
    $|  if (live <= 0 || live > \{rows} || row >= live || group >= \{groups}) return;
    $|  const int partition = threadIdx.x % \{partitions}, column = tile * \{columns} + threadIdx.x / \{partitions};
    $|  const unsigned long long input_base = ((unsigned long long)row * \{groups}ULL + group) * \{input}ULL;
    $|  const int weight_row = group * \{rank} + column;
    $|  const unsigned long long weight_base = (unsigned long long)weight_row * \{input}ULL;
    $|  float sum = 0.0f;
    $|  if (column < \{rank}) for (int inner = partition; inner < \{input}; inner += \{partitions}) {
    $|    const unsigned char bits = weight[weight_base + inner], scale = scales[(unsigned long long)(weight_row / 128) * \{input / 128}ULL + inner / 128];
    $|    if (bits == 0x7fU || bits == 0xffU || scale == 0xffU) { sum = nanf(""); break; }
    $|    const float parameter = __bfloat162float(__float2bfloat16_rn(__fmul_rn(lf_block_e4m3_decode(bits), lf_block_e8m0_decode(scale))));
    $|    sum = __fadd_rn(sum, __fmul_rn(__bfloat162float(input[input_base + inner]), parameter));
    $|  }
    $|  for (int span = \{partitions / 2}; span > 0; span >>= 1) {
    $|    const float upper = __shfl_down_sync(0xffffffffU, sum, span, \{partitions});
    $|    if (partition < span) sum = __fadd_rn(sum, upper);
    $|  }
    $|  if (partition == 0 && column < \{rank}) output[((unsigned long long)row * \{groups}ULL + group) * \{rank}ULL + column] = __float2bfloat16_rn(sum);
    $|}
    $|
  )
}
