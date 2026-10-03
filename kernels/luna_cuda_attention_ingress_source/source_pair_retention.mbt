///|
/// CUDA realization of the middle-end's local split-pair ownership. Initial
/// component loads and the square-sum retain ascending strided order. The
/// normalization BF16 round remains observable even though its value no longer
/// travels through shared scratch before paired rotation.
fn lower_retained_attention_ingress_numerics(
  plan : @luna_projection_tile_compiler.AttentionIngressNumerics,
  retention : @elementwise_physical_ir.SplitPairRetention,
) -> String {
  let (half, owners, pairs) = retention.geometry()
  let epsilon = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(
    plan.epsilon().to_double(),
  )
  let out = StringBuilder()
  out <+ "  float first_values[\{pairs}], second_values[\{pairs}];\n"
  out <+ "  float square_sum = 0.0f;\n"
  // Keep the original fold's component order: all first-half components,
  // followed by the second half. Interleaving pairs would reassociate it.
  out <+ "  #pragma unroll\n"
  out <+ "  for (int item = 0; item < \{pairs}; ++item) {\n"
  out <+ "    const float first = values[lane + item * \{owners}];\n"
  out <+ "    first_values[item] = first;\n"
  out <+
    "    square_sum = __fadd_rn(square_sum, __fmul_rn(first, first));\n  }\n"
  out <+ "  #pragma unroll\n"
  out <+ "  for (int item = 0; item < \{pairs}; ++item) {\n"
  out <+ "    const float second = values[lane + item * \{owners} + \{half}];\n"
  out <+ "    second_values[item] = second;\n"
  out <+
    "    square_sum = __fadd_rn(square_sum, __fmul_rn(second, second));\n  }\n"
  out <+
    "  for (int delta = 16; delta > 0; delta >>= 1) square_sum = __fadd_rn(square_sum, __shfl_down_sync(0xffffffffU, square_sum, delta));\n"
  out <+
    "  const float inverse_rms = __shfl_sync(0xffffffffU, lane == 0 ? rsqrtf(__fadd_rn(__fdividef(square_sum, (float)\{half * 2}), \{epsilon})) : 0.0f, 0);\n"
  out <+ "  #pragma unroll\n"
  out <+ "  for (int item = 0; item < \{pairs}; ++item) {\n"
  out <+ "    const int pair = lane + item * \{owners};\n"
  out <+
    "    const float first = __bfloat162float(__float2bfloat16_rn(__fmul_rn(__fmul_rn(first_values[item], inverse_rms), __bfloat162float(weight[pair]))));\n"
  out <+
    "    const float second = __bfloat162float(__float2bfloat16_rn(__fmul_rn(__fmul_rn(second_values[item], inverse_rms), __bfloat162float(weight[pair + \{half}]))));\n"
  out <+
    "    const float sine = rotary.sine[item], cosine = rotary.cosine[item];\n"
  out <+
    "    values[pair] = __bfloat162float(__float2bfloat16_rn(__fsub_rn(__fmul_rn(first, cosine), __fmul_rn(second, sine))));\n"
  out <+
    "    values[pair + \{half}] = __bfloat162float(__float2bfloat16_rn(__fadd_rn(__fmul_rn(second, cosine), __fmul_rn(first, sine))));\n"
  out <+ "  }\n"
  // Each output pair is written/read by its retained owner. No cross-owner
  // publication is needed; callers still publish scratch before entering.
  out.to_string()
}
