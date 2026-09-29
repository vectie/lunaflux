///|
/// Emits the canary-free production ABI. Keep this source path separate from
/// qualification so production cannot accidentally inherit a diagnostic
/// pointer, atomic write, or host-observation contract.
fn fused_residual_rmsnorm_production_source(
  symbol : String,
  parameters : FusedResidualNormParameters,
  profile : @launch_contract.PagedKernelProfile,
  physical_program : @elementwise_physical_ir.NormalizationProgram,
) -> Bytes raise FusedParallelAotError {
  let (width, owners, _) = physical_program.geometry()
  let output = StringBuilder()
  output.write_string(
    "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n",
  )
  output.write_string("#define LF_BLOCK_THREADS " + owners.to_string() + "\n")
  output.write_string("#define LF_WIDTH " + width.to_string() + "\n")
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
    residual_program_source(physical_program) +
    "}\n",
  )
  @utf8.encode(output.to_string())
}
