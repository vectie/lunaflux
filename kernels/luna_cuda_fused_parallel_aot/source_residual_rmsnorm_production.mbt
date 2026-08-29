///|
/// Emits the canary-free production ABI. Keep this source path separate from
/// qualification so production cannot accidentally inherit a diagnostic
/// pointer, atomic write, or host-observation contract.
fn fused_residual_rmsnorm_production_source(
  symbol : String,
  parameters : FusedResidualNormParameters,
  profile : @launch_contract.PagedKernelProfile,
) -> Bytes {
  let output = StringBuilder()
  output.write_string(
    "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n",
  )
  output.write_string("#define LF_BLOCK_THREADS 128\n")
  output.write_string("#define LF_WIDTH " + parameters.width.to_string() + "\n")
  output.write_string(
    "#define LF_EPSILON " +
    @luna_cuda_pointwise_aot.canonical_cuda_float_literal(parameters.epsilon) +
    "\n",
  )
  output.write_string("extern \"C\" __global__ void " + symbol + "(\n")
  output.write_string(
    "  const int* step_counts, const __nv_bfloat16* residual,\n" +
    "  const __nv_bfloat16* branch, const __nv_bfloat16* norm_weight,\n" +
    "  __nv_bfloat16* residual_output, __nv_bfloat16* norm_output) {\n" +
    "  if (blockDim.x != LF_BLOCK_THREADS || blockIdx.y != 0 || blockIdx.z != 0) return;\n" +
    "  const int token = (int)blockIdx.x;\n" +
    "  const int prefill = step_counts[0], decode = step_counts[1];\n" +
    "  const int rows = step_counts[2], tokens = step_counts[3];\n" +
    "  if (prefill < 0 || decode < 0 || rows <= 0 || rows > " +
    profile.max_query_rows().to_string() +
    " || prefill > rows || decode != rows - prefill ||\n" +
    "      tokens < rows || tokens > " +
    profile.max_query_tokens().to_string() +
    " || token >= tokens) return;\n" +
    "  __shared__ float partial[LF_BLOCK_THREADS];\n" +
    "  float local_sum = 0.0f;\n" +
    "  for (int column = (int)threadIdx.x; column < LF_WIDTH; column += LF_BLOCK_THREADS) {\n" +
    "    const int offset = token * LF_WIDTH + column;\n" +
    "    const float value = __bfloat162float(residual[offset]) + __bfloat162float(branch[offset]);\n" +
    "    const __nv_bfloat16 encoded = __float2bfloat16_rn(value);\n" +
    "    residual_output[offset] = encoded;\n" +
    "    const float rounded = __bfloat162float(encoded);\n" +
    "    local_sum += rounded * rounded;\n" +
    "  }\n" +
    "  partial[threadIdx.x] = local_sum;\n" +
    "  __syncthreads();\n" +
    "  for (int stride = LF_BLOCK_THREADS / 2; stride > 0; stride >>= 1) {\n" +
    "    if ((int)threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];\n" +
    "    __syncthreads();\n" +
    "  }\n" +
    "  const float inverse_rms = rsqrtf(partial[0] / (float)LF_WIDTH + LF_EPSILON);\n" +
    "  for (int column = (int)threadIdx.x; column < LF_WIDTH; column += LF_BLOCK_THREADS) {\n" +
    "    const int offset = token * LF_WIDTH + column;\n" +
    "    const float value = __bfloat162float(residual_output[offset]);\n" +
    "    const float weight = __bfloat162float(norm_weight[column]);\n" +
    "    norm_output[offset] = __float2bfloat16_rn(value * inverse_rms * weight);\n" +
    "  }\n" +
    "}\n",
  )
  @utf8.encode(output.to_string())
}
