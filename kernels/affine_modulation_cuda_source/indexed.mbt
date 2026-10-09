///|
/// Indexed BF16 modulation with compact [parameter_rows,width] tables.
/// Emits prefix_affine(normalized,shift,scale,indices,output) and
/// prefix_residual(residual,gate,branch,indices,output), 256 threads each.
/// Each arithmetic operation rounds through BF16, including 1+scale.
/// Indices must be validated once at startup; no expanded per-token tables.
pub fn indexed_bf16_modulation_source(
  rows~ : Int,
  width~ : Int,
  parameter_rows~ : Int,
  function_prefix~ : String,
) -> Bytes raise AffineModulationCudaSourceError {
  if rows <= 0 || parameter_rows <= 0 {
    raise InvalidRowCount
  }
  if width <= 0 {
    raise InvalidWidth
  }
  if rows.to_int64() > 2147483647L / width.to_int64() ||
    parameter_rows.to_int64() > 2147483647L / width.to_int64() {
    raise DimensionOverflow
  }
  if !valid_symbol(function_prefix) {
    raise InvalidFunctionSymbol
  }
  let extent = (rows.to_int64() * width.to_int64()).to_string()
  let w = width.to_string()
  let addressing = "const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x; if (i >= " +
    extent +
    "ULL) return; const unsigned long long p = (unsigned long long)indices[i / " +
    w +
    "ULL] * " +
    w +
    "ULL + i % " +
    w +
    "ULL;\n"
  @utf8.encode(
    "#include <cuda_bf16.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_prefix +
    "_affine(const __nv_bfloat16 *x, const __nv_bfloat16 *shift, const __nv_bfloat16 *scale, const int32_t *indices, __nv_bfloat16 *output) {\n" +
    addressing +
    "const __nv_bfloat16 s = __float2bfloat16_rn(__fadd_rn(1.0f, __bfloat162float(scale[p])));\n" +
    "const __nv_bfloat16 v = __float2bfloat16_rn(__fmul_rn(__bfloat162float(x[i]), __bfloat162float(s)));\n" +
    "output[i] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(v), __bfloat162float(shift[p]))); }\n" +
    "extern \"C\" __global__ void " +
    function_prefix +
    "_residual(const __nv_bfloat16 *x, const __nv_bfloat16 *gate, const __nv_bfloat16 *branch, const int32_t *indices, __nv_bfloat16 *output) {\n" +
    addressing +
    "const __nv_bfloat16 v = __float2bfloat16_rn(__fmul_rn(__bfloat162float(gate[p]), __bfloat162float(branch[i])));\n" +
    "output[i] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(x[i]), __bfloat162float(v))); }\n",
  )
}
