///|
/// Keep original parallel producers and their rounded values. One publication
/// makes their ordered partials available; every consumer subgroup evaluates
/// the same tree suffix locally. This removes repeated workgroup fences without
/// serializing input loads onto one subgroup or changing reduction order.
fn residual_retained_source(
  plan : @elementwise_physical_ir.RetainedNormalizationTree,
) -> String {
  let (owners, lanes, values) = plan.geometry()
  let sums = owners / lanes
  let retained_values = (values - 1) / sums + 1
  let output = StringBuilder()
  output.write_string(
    (
      $|  {
      $|    const int lane = (int)threadIdx.x % \{lanes};
      $|    __shared__ float published[\{owners}];
      $|    float local_sum = 0.0f;
      $|    float retained[\{retained_values}];
      $|    #pragma unroll
      $|    for (int item = 0; item < \{retained_values}; ++item) {
      $|      const int column = (int)threadIdx.x + item * \{owners};
      $|      if (column < LF_WIDTH) {
      $|        const int offset = token * LF_WIDTH + column;
      $|        const float value = __bfloat162float(residual[offset]) + __bfloat162float(branch[offset]);
      $|        const __nv_bfloat16 encoded = __float2bfloat16_rn(value);
      $|        residual_output[offset] = encoded;
      $|        const float rounded = __bfloat162float(encoded);
      $|        retained[item] = rounded;
      $|        local_sum += rounded * rounded;
      $|      }
      $|    }
      $|    published[threadIdx.x] = local_sum;
      $|    __syncthreads();
      $|    float partial[\{sums}];
      $|    #pragma unroll
      $|    for (int group = 0; group < \{sums}; ++group) partial[group] = published[lane + group * \{lanes}];
    ),
  )
  for (destination, source) in plan.local_edges() {
    output.write_string("    partial[\{destination}] += partial[\{source}];\n")
  }
  output.write_string("    float total = partial[0];\n")
  for stride in plan.exchange_strides() {
    output.write_string(
      "    const float peer\{stride} = __shfl_down_sync(0xffffffffU, total, \{stride});\n" +
      "    if (lane < \{stride}) total += peer\{stride};\n",
    )
  }
  output.write_string(
    (
      $|    total = __shfl_sync(0xffffffffU, total, 0);
      $|    const float inverse_rms = rsqrtf(total / (float)LF_WIDTH + LF_EPSILON);
      $|    #pragma unroll
      $|    for (int item = 0; item < \{retained_values}; ++item) {
      $|      const int column = (int)threadIdx.x + item * \{owners};
      $|      if (column < LF_WIDTH) {
      $|        const float weight = __bfloat162float(norm_weight[column]);
      $|        norm_output[token * LF_WIDTH + column] = __float2bfloat16_rn(retained[item] * inverse_rms * weight);
      $|      }
      $|    }
      $|  }
      $|
    ),
  )
  output.to_string()
}
