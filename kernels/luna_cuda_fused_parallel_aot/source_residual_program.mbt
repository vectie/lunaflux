///|
/// Interprets the retained numeric/effect plan during AOT emission. Ownership
/// may change, but the logical tree and F32 order cannot. Wide rows retain the
/// shared realization outside the bounded retained-fragment budget.
fn residual_program_source(
  program : @elementwise_physical_ir.NormalizationProgram,
) -> String raise FusedParallelAotError {
  if program.retained_tree(maximum_values_per_lane=64) is Some(retained) &&
    retained.geometry().1 == 32 {
    return residual_retained_source(retained)
  }
  residual_shared_tree_source(program)
}

///|
fn residual_shared_tree_source(
  program : @elementwise_physical_ir.NormalizationProgram,
) -> String raise FusedParallelAotError {
  let output = StringBuilder()
  output.write_string(
    "  __shared__ float partial[LF_BLOCK_THREADS];\n" +
    "  float local_sum = 0.0f;\n" +
    "  for (int column = (int)threadIdx.x; column < LF_WIDTH; column += LF_BLOCK_THREADS) {\n",
  )
  for step in program.preparation() {
    match step {
      LoadInput =>
        output.write_string(
          "    const int offset = token * LF_WIDTH + column;\n",
        )
      AddResidual =>
        output.write_string(
          "    const float value = __bfloat162float(residual[offset]) + __bfloat162float(branch[offset]);\n",
        )
      RoundBf16 =>
        output.write_string(
          "    const __nv_bfloat16 encoded = __float2bfloat16_rn(value);\n" +
          "    residual_output[offset] = encoded;\n" +
          "    const float rounded = __bfloat162float(encoded);\n",
        )
      AccumulateSquare =>
        output.write_string("    local_sum += rounded * rounded;\n")
    }
  }
  output.write_string("  }\n")
  for effect in program.reduction() {
    match effect {
      PublishPartials =>
        output.write_string("  partial[threadIdx.x] = local_sum;\n")
      SynchronizeOwners => output.write_string("  __syncthreads();\n")
      ReduceOwnerTree(initial_stride) => {
        let (_, owners, _) = program.geometry()
        guard initial_stride == owners / 2 else {
          raise FusedParallelUnsupported(StaticResources)
        }
        output.write_string(
          "  for (int stride = LF_BLOCK_THREADS / 2; stride > 0; stride >>= 1) {\n" +
          "    if ((int)threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];\n" +
          "    __syncthreads();\n" +
          "  }\n",
        )
      }
      ReduceSubgroup | ReducePartials | PublishTotal =>
        raise FusedParallelUnsupported(StaticResources)
    }
  }
  output.write_string(
    "  const float inverse_rms = rsqrtf(partial[0] / (float)LF_WIDTH + LF_EPSILON);\n" +
    "  for (int column = (int)threadIdx.x; column < LF_WIDTH; column += LF_BLOCK_THREADS) {\n" +
    "    const int offset = token * LF_WIDTH + column;\n" +
    "    const float value = __bfloat162float(residual_output[offset]);\n" +
    "    const float weight = __bfloat162float(norm_weight[column]);\n" +
    "    norm_output[offset] = __float2bfloat16_rn(value * inverse_rms * weight);\n" +
    "  }\n",
  )
  output.to_string()
}
