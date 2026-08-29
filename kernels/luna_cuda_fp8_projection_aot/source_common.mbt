///|
fn fp8_cuda_prelude() -> String {
  "#include <cuda_bf16.h>\n" +
  "#include <cuda_fp8.h>\n" +
  "#include <math.h>\n#include <stdint.h>\n" +
  "#if !defined(CUDA_VERSION) || CUDA_VERSION < 12000\n" +
  "#error LunaFlux finite-E4M3 AOT requires CUDA 12 cuda_fp8.h\n#endif\n" +
  "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ != 890) && (__CUDA_ARCH__ != 900)\n" +
  "#error LunaFlux FP8 release v1 is closed to sm_89 and sm_90\n#endif\n" +
  "__device__ __forceinline__ float lf_e4m3_decode(const unsigned char bits) {\n" +
  "  __nv_fp8_e4m3 value; value.__x = bits; return (float)value;\n}\n" +
  "__device__ __forceinline__ float lf_e4m3_round(const float value) {\n" +
  "  const float bounded = fminf(448.0f, fmaxf(-448.0f, value));\n" +
  "  const float canonical = bounded == 0.0f ? 0.0f : bounded;\n" +
  "  const __nv_fp8_storage_t bits = __nv_cvt_float_to_fp8(canonical, __NV_SATFINITE, __NV_E4M3);\n" +
  "  return lf_e4m3_decode(bits);\n}\n" +
  "__device__ __forceinline__ float lf_dynamic_scale(const __nv_bfloat16 *input, const unsigned long long count) {\n" +
  "  float maximum = 0.0f;\n" +
  "  for (unsigned long long i = 0; i < count; ++i) maximum = fmaxf(maximum, fabsf(__bfloat162float(input[i])));\n" +
  "  return maximum == 0.0f ? 1.0f : __fdiv_rn(maximum, 448.0f);\n}\n"
}

///|
fn fp8_count_guard(maximum_tokens : Int) -> String {
  "  const int token_count = counts[3];\n" +
  "  if (token_count <= 0 || token_count > " +
  maximum_tokens.to_string() +
  ") return;\n"
}

///|
fn fp8_common_index(parameters : Fp8ProjectionParameters) -> String {
  "  const unsigned long long index = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;\n" +
  "  const unsigned long long live = (unsigned long long)token_count * " +
  parameters.output_width.to_string() +
  "ULL;\n" +
  "  if (index >= live) return;\n" +
  "  const unsigned long long row = index / " +
  parameters.output_width.to_string() +
  "ULL;\n" +
  "  const int column = (int)(index % " +
  parameters.output_width.to_string() +
  "ULL);\n" +
  "  const unsigned long long input_base = row * " +
  parameters.input_width.to_string() +
  "ULL;\n" +
  "  const float activation_scale = lf_dynamic_scale(input, (unsigned long long)token_count * " +
  parameters.input_width.to_string() +
  "ULL);\n" +
  "  if (index == 0ULL) *activation_scale_workspace = activation_scale;\n"
}

///|
fn fp8_dot_loop(input_width : Int) -> String {
  "  float accumulator = 0.0f;\n" +
  "  for (int inner = 0; inner < " +
  input_width.to_string() +
  "; ++inner) {\n" +
  "    const float input_value = __bfloat162float(input[input_base + inner]);\n" +
  "    const float activation = __fmul_rn(lf_e4m3_round(__fdiv_rn(input_value, activation_scale)), activation_scale);\n" +
  "    const float weight_value = __fmul_rn(lf_e4m3_decode(weight[weight_base + inner]), weight_scale[0]);\n" +
  "    accumulator = __fadd_rn(accumulator, __fmul_rn(activation, weight_value));\n" +
  "  }\n" +
  "  output[index] = __float2bfloat16_rn(accumulator);\n}\n"
}
