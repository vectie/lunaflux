///|
/// Shared numeric decoding vocabulary used by composed projection consumers.
/// The scalar-default bytes are preserved exactly.
fn projection_helpers() -> String {
  "#include <cuda.h>\n#include <cuda_bf16.h>\n#include <cuda_fp8.h>\n#include <math.h>\n#include <stdint.h>\n" +
  "#if !defined(CUDA_VERSION) || CUDA_VERSION < 12080\n#error LunaFlux block FP8 UE8M0 source requires CUDA 12.8 or newer\n#endif\n" +
  "#ifndef LUNAFLUX_BLOCK_FP8_UE8M0_HELPERS_V1\n#define LUNAFLUX_BLOCK_FP8_UE8M0_HELPERS_V1\n" +
  "__device__ __forceinline__ float lf_block_e4m3_decode(const unsigned char bits) { __nv_fp8_e4m3 value; value.__x = bits; return (float)value; }\n" +
  "__device__ __forceinline__ float lf_block_e8m0_decode(const unsigned char bits) { __nv_fp8_e8m0 value; value.__x = bits; return (float)value; }\n" +
  "__device__ __forceinline__ float lf_block_e4m3_round(const float value) { const float bounded = fminf(448.0f, fmaxf(-448.0f, value)); const float canonical = bounded == 0.0f ? 0.0f : bounded; const __nv_fp8_storage_t bits = __nv_cvt_float_to_fp8(canonical, __NV_SATFINITE, __NV_E4M3); return lf_block_e4m3_decode(bits); }\n" +
  "__device__ __forceinline__ float lf_block_activation_scale(const __nv_bfloat16 *input, const unsigned long long base) { float maximum = 0.0001f; for (int inner = 0; inner < 128; ++inner) maximum = fmaxf(maximum, fabsf(__bfloat162float(input[base + (unsigned long long)inner]))); const float requested = __fmul_rn(maximum, 0.002232142857142857f); const unsigned int bits = __float_as_uint(requested); const int exponent = (int)((bits >> 23) & 0xffU) - 127 + ((bits & 0x7fffffU) != 0U ? 1 : 0); return __uint_as_float((unsigned int)(exponent + 127) << 23); }\n" +
  "#endif\n"
}
