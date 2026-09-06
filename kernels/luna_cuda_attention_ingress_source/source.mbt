///|
/// Terminal CUDA lowering of the same pure epilogue used on both sides of a
/// fusion cut. A complete subgroup owns one head; callers own storage, row
/// traversal and the effectful KV commit. No barrier may involve another head.
pub fn lower_attention_ingress_numerics(
  plan : @luna_projection_tile_compiler.AttentionIngressNumerics,
) -> String {
  let (dimension, lanes) = plan.shape()
  guard lanes == 32 else {
    abort("CUDA attention ingress requires a 32-lane numerical schedule")
  }
  let out = StringBuilder()
  out <+
    "static __device__ __constant__ float LF_ROPE_INV_FREQ[\{dimension / 2}] = {"
  for index, frequency in plan.rotary_basis() {
    if index > 0 {
      out.write_string(", ")
    }
    out.write_string(
      @luna_cuda_pointwise_aot.canonical_cuda_float_literal(
        frequency.to_double(),
      ),
    )
  }
  out.write_string("};\n")
  out <+
    "static __device__ __forceinline__ void lf_ingress_norm_rotary(float* values, const __nv_bfloat16* weight, int position, int lane) {\n"
  out <+ "  float square_sum = 0.0f;\n"
  out <+
    "  for (int component = lane; component < \{dimension}; component += 32) square_sum = __fadd_rn(square_sum, __fmul_rn(values[component], values[component]));\n"
  out <+
    "  for (int delta = 16; delta > 0; delta >>= 1) square_sum = __fadd_rn(square_sum, __shfl_down_sync(0xffffffffU, square_sum, delta));\n"
  let epsilon = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(
    plan.epsilon().to_double(),
  )
  out <+
    "  const float inverse_rms = __shfl_sync(0xffffffffU, lane == 0 ? rsqrtf(__fadd_rn(__fdividef(square_sum, (float)\{dimension}), \{epsilon})) : 0.0f, 0);\n"
  out <+
    "  for (int component = lane; component < \{dimension}; component += 32) values[component] = __bfloat162float(__float2bfloat16_rn(__fmul_rn(__fmul_rn(values[component], inverse_rms), __bfloat162float(weight[component]))));\n"
  out <+ "  __syncwarp();\n"
  out <+ "  for (int pair = lane; pair < \{dimension / 2}; pair += 32) {\n"
  out <+
    "    const float first = values[pair], second = values[pair + \{dimension / 2}];\n"
  out <+ "    float sine, cosine;\n"
  out <+
    "    sincosf(__fmul_rn((float)position, LF_ROPE_INV_FREQ[pair]), &sine, &cosine);\n"
  out <+
    "    values[pair] = __bfloat162float(__float2bfloat16_rn(__fsub_rn(__fmul_rn(first, cosine), __fmul_rn(second, sine))));\n"
  out <+
    "    values[pair + \{dimension / 2}] = __bfloat162float(__float2bfloat16_rn(__fadd_rn(__fmul_rn(second, cosine), __fmul_rn(first, sine))));\n"
  out <+ "  }\n  __syncwarp();\n}\n"
  out.to_string()
}
