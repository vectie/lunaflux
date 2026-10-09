///|
fn valid_symbol(symbol : String) -> Bool {
  if symbol.length() == 0 || symbol.length() > 128 {
    return false
  }
  for index, character in symbol {
    if index == 0 {
      if !(character is ('a'..='z' | 'A'..='Z' | '_')) {
        return false
      }
    } else if !(character is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_')) {
      return false
    }
  }
  true
}

///|
pub fn block_fp8_ue8m0_projection_cuda_source(
  input_width~ : Int,
  output_width~ : Int,
  maximum_rows~ : Int,
  function_symbol~ : String,
) -> BlockFp8Ue8M0ProjectionCudaSource raise BlockFp8Ue8M0ProjectionCudaSourceError {
  let block_width = 128
  if input_width <= 0 ||
    input_width > 65536 ||
    output_width <= 0 ||
    output_width > 65536 ||
    maximum_rows <= 0 ||
    maximum_rows > 65536 ||
    input_width % block_width != 0 ||
    output_width % block_width != 0 {
    raise InvalidGeometry
  }
  if !valid_symbol(function_symbol) {
    raise InvalidSymbol
  }
  let live = maximum_rows.to_int64() * output_width.to_int64()
  if live <= 0L || live > 2147483647L {
    raise SizeOverflow
  }
  let source = @utf8.encode(
    "#include <cuda.h>\n#include <cuda_bf16.h>\n#include <cuda_fp8.h>\n#include <math.h>\n#include <stdint.h>\n" +
    "#if !defined(CUDA_VERSION) || CUDA_VERSION < 12080\n#error LunaFlux block FP8 UE8M0 source requires CUDA 12.8 or newer\n#endif\n" +
    "#ifndef LUNAFLUX_BLOCK_FP8_UE8M0_HELPERS_V1\n#define LUNAFLUX_BLOCK_FP8_UE8M0_HELPERS_V1\n" +
    "__device__ __forceinline__ float lf_block_e4m3_decode(const unsigned char bits) { __nv_fp8_e4m3 value; value.__x = bits; return (float)value; }\n" +
    "__device__ __forceinline__ float lf_block_e8m0_decode(const unsigned char bits) { __nv_fp8_e8m0 value; value.__x = bits; return (float)value; }\n" +
    "__device__ __forceinline__ float lf_block_e4m3_round(const float value) { const float bounded = fminf(448.0f, fmaxf(-448.0f, value)); const float canonical = bounded == 0.0f ? 0.0f : bounded; const __nv_fp8_storage_t bits = __nv_cvt_float_to_fp8(canonical, __NV_SATFINITE, __NV_E4M3); return lf_block_e4m3_decode(bits); }\n" +
    "__device__ __forceinline__ float lf_block_activation_scale(const __nv_bfloat16 *input, const unsigned long long base) { float maximum = 0.0001f; for (int inner = 0; inner < 128; ++inner) maximum = fmaxf(maximum, fabsf(__bfloat162float(input[base + (unsigned long long)inner]))); const float requested = __fmul_rn(maximum, 0.002232142857142857f); const unsigned int bits = __float_as_uint(requested); const int exponent = (int)((bits >> 23) & 0xffU) - 127 + ((bits & 0x7fffffU) != 0U ? 1 : 0); return __uint_as_float((unsigned int)(exponent + 127) << 23); }\n" +
    "#endif\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int *counts, const __nv_bfloat16 *input, const unsigned char *weight, const unsigned char *weight_scale, __nv_bfloat16 *output) {\n" +
    "  const int row_count = counts[3]; if (row_count <= 0 || row_count > " +
    maximum_rows.to_string() +
    ") return;\n" +
    "  const unsigned long long row = (unsigned long long)blockIdx.x; if (row >= (unsigned long long)row_count) return; const int column = (int)blockIdx.y * 256 + threadIdx.x; if (column >= " +
    output_width.to_string() +
    ") return; const unsigned long long index = row * " +
    output_width.to_string() +
    "ULL + (unsigned long long)column; const unsigned long long input_base = row * " +
    input_width.to_string() +
    "ULL; const unsigned long long weight_base = (unsigned long long)column * " +
    input_width.to_string() +
    "ULL; float accumulator = 0.0f;\n" +
    "  for (int block = 0; block < " +
    (input_width / block_width).to_string() +
    "; ++block) { const unsigned long long block_base = (unsigned long long)block * 128ULL; const float activation_scale = lf_block_activation_scale(input, input_base + block_base); const unsigned long long scale_index = (unsigned long long)(column / 128) * " +
    (input_width / block_width).to_string() +
    "ULL + (unsigned long long)block; const unsigned char weight_scale_bits = weight_scale[scale_index]; if (weight_scale_bits == 0xffU) return; const float parameter_scale = lf_block_e8m0_decode(weight_scale_bits); float block_accumulator = 0.0f;\n" +
    "    for (int offset = 0; offset < 128; ++offset) { const unsigned long long inner = block_base + (unsigned long long)offset; const float raw = __bfloat162float(input[input_base + inner]); const float activation = lf_block_e4m3_round(__fdiv_rn(raw, activation_scale)); const unsigned char weight_bits = weight[weight_base + inner]; if (weight_bits == 0x7fU || weight_bits == 0xffU) return; const float parameter = lf_block_e4m3_decode(weight_bits); block_accumulator = __fadd_rn(block_accumulator, __fmul_rn(activation, parameter)); }\n" +
    "    const float block_scale = __fmul_rn(activation_scale, parameter_scale); accumulator = __fadd_rn(accumulator, __fmul_rn(block_accumulator, block_scale)); }\n" +
    "  output[index] = __float2bfloat16_rn(accumulator);\n}\n",
  )
  {
    source,
    input_width,
    output_width,
    maximum_rows,
    block_width,
    function_symbol,
  }
}
