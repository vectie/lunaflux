///|
fn fp8_cuda_prelude_v2() -> String {
  "#include <cuda_bf16.h>\n" +
  "#include <cuda_fp8.h>\n" +
  "#include <math.h>\n#include <stdint.h>\n" +
  "#if !defined(CUDA_VERSION) || CUDA_VERSION < 12000\n" +
  "#error LunaFlux finite-E4M3 AOT v2 requires CUDA 12 cuda_fp8.h\n#endif\n" +
  "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ != 890) && (__CUDA_ARCH__ != 900) && (__CUDA_ARCH__ != 1200)\n" +
  "#error LunaFlux FP8 release v2 is closed to sm_89, sm_90, and sm_120\n#endif\n" +
  "__device__ __forceinline__ float lf_v2_e4m3_decode(const unsigned char bits) {\n" +
  "  __nv_fp8_e4m3 value; value.__x = bits; return (float)value;\n}\n" +
  "__device__ __forceinline__ float lf_v2_e4m3_round(const float value) {\n" +
  "  const float bounded = fminf(448.0f, fmaxf(-448.0f, value));\n" +
  "  const float canonical = bounded == 0.0f ? 0.0f : bounded;\n" +
  "  const __nv_fp8_storage_t bits = __nv_cvt_float_to_fp8(canonical, __NV_SATFINITE, __NV_E4M3);\n" +
  "  return lf_v2_e4m3_decode(bits);\n}\n" +
  "__device__ __forceinline__ float lf_v2_failure(void) { return __uint_as_float(0x7fc00000U); }\n" +
  "__device__ __forceinline__ bool lf_v2_valid_scale(const float value) { return isfinite(value) && value > 0.0f; }\n" +
  "__device__ __forceinline__ float lf_v2_dynamic_scale(const __nv_bfloat16 *input, const unsigned long long count) {\n" +
  "  float maximum = 0.0f;\n" +
  "  for (unsigned long long i = 0; i < count; ++i) {\n" +
  "    const float value = __bfloat162float(input[i]);\n" +
  "    if (!isfinite(value)) return lf_v2_failure();\n" +
  "    maximum = fmaxf(maximum, fabsf(value));\n  }\n" +
  "  const float scale = maximum == 0.0f ? 1.0f : __fdiv_rn(maximum, 448.0f);\n" +
  "  return lf_v2_valid_scale(scale) ? scale : lf_v2_failure();\n}\n"
}

///|
fn fp8_v2_serial_entry(maximum_tokens : Int, workspace_cells : Int) -> String {
  "  const unsigned long long index = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;\n" +
  "  if (index != 0ULL) return;\n" +
  "  workspace[0] = lf_v2_failure();\n" +
  (if workspace_cells == 2 {
    "  workspace[1] = lf_v2_failure();\n"
  } else {
    ""
  }) +
  "  const int token_count = counts[3];\n" +
  "  if (token_count <= 0 || token_count > " +
  maximum_tokens.to_string() +
  ") return;\n"
}

///|
fn fp8_dense_source_v2(
  symbol : String,
  parameters : Fp8ProjectionParameters,
  maximum_tokens : Int,
) -> Bytes {
  let input_width = parameters.input_width.to_string()
  let output_width = parameters.output_width.to_string()
  @utf8.encode(
    fp8_cuda_prelude_v2() +
    "__device__ __forceinline__ float lf_v2_dense_value(const __nv_bfloat16 *input, const unsigned long long row, const int column, const float external_scale, const unsigned char *weight, const float weight_scale) {\n" +
    "  if (!lf_v2_valid_scale(external_scale) || !lf_v2_valid_scale(weight_scale)) return lf_v2_failure();\n" +
    "  const unsigned long long input_base = row * " +
    input_width +
    "ULL; const unsigned long long weight_base = (unsigned long long)column * " +
    input_width +
    "ULL;\n  float accumulator = 0.0f;\n" +
    "  for (int inner = 0; inner < " +
    input_width +
    "; ++inner) {\n" +
    "    const float raw = __bfloat162float(input[input_base + inner]);\n" +
    "    const float activation = __fmul_rn(lf_v2_e4m3_round(__fdiv_rn(raw, external_scale)), external_scale);\n" +
    "    const float weight_value = __fmul_rn(lf_v2_e4m3_decode(weight[weight_base + inner]), weight_scale);\n" +
    "    if (!isfinite(raw) || !isfinite(activation) || !isfinite(weight_value)) return lf_v2_failure();\n" +
    "    accumulator = __fadd_rn(accumulator, __fmul_rn(activation, weight_value));\n" +
    "    if (!isfinite(accumulator)) return lf_v2_failure();\n  }\n" +
    "  const __nv_bfloat16 converted = __float2bfloat16_rn(accumulator);\n" +
    "  return isfinite(__bfloat162float(converted)) ? accumulator : lf_v2_failure();\n}\n" +
    "extern \"C\" __global__ void " +
    symbol +
    "(const int32_t *counts, const __nv_bfloat16 *input, " +
    "const unsigned char *weight, const float *weight_scale, " +
    "__nv_bfloat16 *output, float *workspace) {\n" +
    fp8_v2_serial_entry(maximum_tokens, 1) +
    "  const float external_scale = lf_v2_dynamic_scale(input, (unsigned long long)token_count * " +
    input_width +
    "ULL);\n  if (!lf_v2_valid_scale(external_scale)) return;\n" +
    "  const unsigned long long live = (unsigned long long)token_count * " +
    output_width +
    "ULL;\n" +
    "  for (unsigned long long output_index = 0; output_index < live; ++output_index) {\n" +
    "    const unsigned long long row = output_index / " +
    output_width +
    "ULL; const int column = (int)(output_index % " +
    output_width +
    "ULL);\n" +
    "    if (!isfinite(lf_v2_dense_value(input, row, column, external_scale, weight, weight_scale[0]))) return;\n  }\n" +
    "  for (unsigned long long output_index = 0; output_index < live; ++output_index) {\n" +
    "    const unsigned long long row = output_index / " +
    output_width +
    "ULL; const int column = (int)(output_index % " +
    output_width +
    "ULL);\n" +
    "    output[output_index] = __float2bfloat16_rn(lf_v2_dense_value(input, row, column, external_scale, weight, weight_scale[0]));\n  }\n" +
    "  workspace[0] = external_scale;\n}\n",
  )
}

///|
fn fp8_qkv_source_v2(
  symbol : String,
  parameters : Fp8ProjectionParameters,
  maximum_tokens : Int,
) -> Bytes {
  let input_width = parameters.input_width.to_string()
  let output_width = parameters.output_width.to_string()
  let kv_width = parameters.intermediate_width.to_string()
  @utf8.encode(
    fp8_cuda_prelude_v2() +
    "__device__ __forceinline__ float lf_v2_qkv_value(const __nv_bfloat16 *input, const unsigned long long row, const int column, const float external_scale, const unsigned char *query_weight, const float query_scale, const unsigned char *key_weight, const float key_scale, const unsigned char *value_weight, const float value_scale) {\n" +
    "  const unsigned char *weight = query_weight; float weight_scale = query_scale; int local_column = column;\n" +
    "  if (column >= " +
    input_width +
    ") { local_column = column - " +
    input_width +
    "; if (local_column >= " +
    kv_width +
    ") { weight = value_weight; weight_scale = value_scale; local_column -= " +
    kv_width +
    "; } else { weight = key_weight; weight_scale = key_scale; } }\n" +
    "  if (!lf_v2_valid_scale(external_scale) || !lf_v2_valid_scale(weight_scale)) return lf_v2_failure();\n" +
    "  const unsigned long long input_base = row * " +
    input_width +
    "ULL; const unsigned long long weight_base = (unsigned long long)local_column * " +
    input_width +
    "ULL; float accumulator = 0.0f;\n" +
    "  for (int inner = 0; inner < " +
    input_width +
    "; ++inner) { const float raw = __bfloat162float(input[input_base + inner]); const float activation = __fmul_rn(lf_v2_e4m3_round(__fdiv_rn(raw, external_scale)), external_scale); const float weight_value = __fmul_rn(lf_v2_e4m3_decode(weight[weight_base + inner]), weight_scale); if (!isfinite(raw) || !isfinite(activation) || !isfinite(weight_value)) return lf_v2_failure(); accumulator = __fadd_rn(accumulator, __fmul_rn(activation, weight_value)); if (!isfinite(accumulator)) return lf_v2_failure(); }\n" +
    "  const __nv_bfloat16 converted = __float2bfloat16_rn(accumulator); return isfinite(__bfloat162float(converted)) ? accumulator : lf_v2_failure();\n}\n" +
    "extern \"C\" __global__ void " +
    symbol +
    "(const int32_t *counts, const __nv_bfloat16 *input, " +
    "const unsigned char *query_weight, const float *query_scale, " +
    "const unsigned char *key_weight, const float *key_scale, " +
    "const unsigned char *value_weight, const float *value_scale, " +
    "__nv_bfloat16 *output, float *workspace) {\n" +
    fp8_v2_serial_entry(maximum_tokens, 1) +
    "  const float external_scale = lf_v2_dynamic_scale(input, (unsigned long long)token_count * " +
    input_width +
    "ULL); if (!lf_v2_valid_scale(external_scale)) return;\n" +
    "  const unsigned long long live = (unsigned long long)token_count * " +
    output_width +
    "ULL;\n" +
    "  for (unsigned long long output_index = 0; output_index < live; ++output_index) { const unsigned long long row = output_index / " +
    output_width +
    "ULL; const int column = (int)(output_index % " +
    output_width +
    "ULL); if (!isfinite(lf_v2_qkv_value(input, row, column, external_scale, query_weight, query_scale[0], key_weight, key_scale[0], value_weight, value_scale[0]))) return; }\n" +
    "  for (unsigned long long output_index = 0; output_index < live; ++output_index) { const unsigned long long row = output_index / " +
    output_width +
    "ULL; const int column = (int)(output_index % " +
    output_width +
    "ULL); output[output_index] = __float2bfloat16_rn(lf_v2_qkv_value(input, row, column, external_scale, query_weight, query_scale[0], key_weight, key_scale[0], value_weight, value_scale[0])); }\n" +
    "  workspace[0] = external_scale;\n}\n",
  )
}

///|
fn fp8_gated_source_v2(
  symbol : String,
  parameters : Fp8ProjectionParameters,
  maximum_tokens : Int,
) -> Bytes {
  let hidden = parameters.input_width.to_string()
  let intermediate = parameters.intermediate_width.to_string()
  @utf8.encode(
    fp8_cuda_prelude_v2() +
    "__device__ __forceinline__ float lf_v2_gated_z(const __nv_bfloat16 *input, const unsigned long long row, const int column, const float external_scale, const unsigned char *gate_weight, const float gate_scale, const unsigned char *up_weight, const float up_scale) {\n" +
    "  if (!lf_v2_valid_scale(external_scale) || !lf_v2_valid_scale(gate_scale) || !lf_v2_valid_scale(up_scale)) return lf_v2_failure();\n" +
    "  const unsigned long long input_base = row * " +
    hidden +
    "ULL;\n" +
    "  const unsigned long long weight_base = (unsigned long long)column * " +
    hidden +
    "ULL;\n" +
    "  float gate = 0.0f; float up = 0.0f;\n" +
    "  for (int inner = 0; inner < " +
    hidden +
    "; ++inner) {\n" +
    "    const float raw = __bfloat162float(input[input_base + inner]);\n" +
    "    const float activation = __fmul_rn(lf_v2_e4m3_round(__fdiv_rn(raw, external_scale)), external_scale);\n" +
    "    const float gate_value = __fmul_rn(lf_v2_e4m3_decode(gate_weight[weight_base + inner]), gate_scale);\n" +
    "    const float up_value = __fmul_rn(lf_v2_e4m3_decode(up_weight[weight_base + inner]), up_scale);\n" +
    "    if (!isfinite(raw) || !isfinite(activation) || !isfinite(gate_value) || !isfinite(up_value)) return lf_v2_failure();\n" +
    "    gate = __fadd_rn(gate, __fmul_rn(activation, gate_value));\n" +
    "    up = __fadd_rn(up, __fmul_rn(activation, up_value));\n" +
    "    if (!isfinite(gate) || !isfinite(up)) return lf_v2_failure();\n" +
    "  }\n" +
    "  const float bound_expf = expf(-gate);\n" +
    "  if (!isfinite(bound_expf)) return lf_v2_failure();\n" +
    "  const float denominator = __fadd_rn(1.0f, bound_expf);\n" +
    "  const float silu = __fdiv_rn(gate, denominator);\n" +
    "  const float z = __fmul_rn(silu, up);\n" +
    "  return isfinite(denominator) && isfinite(silu) && isfinite(z) ? z : lf_v2_failure();\n}\n" +
    "__device__ __forceinline__ float lf_v2_gated_down_value(const __nv_bfloat16 *input, const unsigned long long row, const int column, const float external_scale, const float post_silu_scale, const unsigned char *gate_weight, const float gate_scale, const unsigned char *up_weight, const float up_scale, const unsigned char *down_weight, const float down_scale) {\n" +
    "  if (!lf_v2_valid_scale(post_silu_scale) || !lf_v2_valid_scale(down_scale)) return lf_v2_failure();\n" +
    "  const unsigned long long down_base = (unsigned long long)column * " +
    intermediate +
    "ULL; float accumulator = 0.0f;\n" +
    "  for (int inner = 0; inner < " +
    intermediate +
    "; ++inner) { const float z = lf_v2_gated_z(input, row, inner, external_scale, gate_weight, gate_scale, up_weight, up_scale); const float activation = __fmul_rn(lf_v2_e4m3_round(__fdiv_rn(z, post_silu_scale)), post_silu_scale); const float weight_value = __fmul_rn(lf_v2_e4m3_decode(down_weight[down_base + inner]), down_scale); if (!isfinite(z) || !isfinite(activation) || !isfinite(weight_value)) return lf_v2_failure(); accumulator = __fadd_rn(accumulator, __fmul_rn(activation, weight_value)); if (!isfinite(accumulator)) return lf_v2_failure(); }\n" +
    "  const __nv_bfloat16 converted = __float2bfloat16_rn(accumulator); return isfinite(__bfloat162float(converted)) ? accumulator : lf_v2_failure();\n}\n" +
    "extern \"C\" __global__ void " +
    symbol +
    "(const int32_t *counts, const __nv_bfloat16 *input, " +
    "const unsigned char *gate_weight, const float *gate_scale, " +
    "const unsigned char *up_weight, const float *up_scale, " +
    "const unsigned char *down_weight, const float *down_scale, " +
    "__nv_bfloat16 *output, float *workspace) {\n" +
    fp8_v2_serial_entry(maximum_tokens, 2) +
    "  const unsigned long long live = (unsigned long long)token_count * " +
    hidden +
    "ULL;\n" +
    "  const float external_scale = lf_v2_dynamic_scale(input, (unsigned long long)token_count * " +
    hidden +
    "ULL);\n" +
    "  if (!lf_v2_valid_scale(external_scale)) return;\n" +
    "  float post_silu_maximum = 0.0f;\n" +
    "  for (int scale_row = 0; scale_row < token_count; ++scale_row) {\n" +
    "    for (int scale_column = 0; scale_column < " +
    intermediate +
    "; ++scale_column) {\n" +
    "      const float z = lf_v2_gated_z(input, (unsigned long long)scale_row, scale_column, external_scale, gate_weight, gate_scale[0], up_weight, up_scale[0]);\n" +
    "      if (!isfinite(z)) return;\n" +
    "      post_silu_maximum = fmaxf(post_silu_maximum, fabsf(z));\n" +
    "    }\n" +
    "  }\n" +
    "  const float post_silu_scale = post_silu_maximum == 0.0f ? 1.0f : __fdiv_rn(post_silu_maximum, 448.0f);\n" +
    "  if (!lf_v2_valid_scale(post_silu_scale)) return;\n" +
    "  for (unsigned long long output_index = 0; output_index < live; ++output_index) { const unsigned long long row = output_index / " +
    hidden +
    "ULL; const int column = (int)(output_index % " +
    hidden +
    "ULL); if (!isfinite(lf_v2_gated_down_value(input, row, column, external_scale, post_silu_scale, gate_weight, gate_scale[0], up_weight, up_scale[0], down_weight, down_scale[0]))) return; }\n" +
    "  for (unsigned long long output_index = 0; output_index < live; ++output_index) { const unsigned long long row = output_index / " +
    hidden +
    "ULL; const int column = (int)(output_index % " +
    hidden +
    "ULL); output[output_index] = __float2bfloat16_rn(lf_v2_gated_down_value(input, row, column, external_scale, post_silu_scale, gate_weight, gate_scale[0], up_weight, up_scale[0], down_weight, down_scale[0])); }\n" +
    "  workspace[0] = external_scale; workspace[1] = post_silu_scale;\n}\n",
  )
}

///|
fn fp8_projection_source_v2(
  symbol : String,
  parameters : Fp8ProjectionParameters,
  maximum_tokens : Int,
) -> Bytes {
  match parameters.family {
    QkvProjection => fp8_qkv_source_v2(symbol, parameters, maximum_tokens)
    DenseProjection | LanguageModelHead =>
      fp8_dense_source_v2(symbol, parameters, maximum_tokens)
    GatedMlp => fp8_gated_source_v2(symbol, parameters, maximum_tokens)
  }
}
