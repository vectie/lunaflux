///|
/// Final row-local modulation using compact timestep tables. Input and table
/// arithmetic retain BF16 rounding; only the final value is widened to F32.
/// Pass a contiguous selected slice of normalized rows and its index map to
/// avoid projecting text, conditioning or padding through a media output head.
pub fn indexed_bf16_shift_scale_f32_source(
  rows~ : Int,
  width~ : Int,
  parameter_rows~ : Int,
  function_symbol~ : String,
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
  if !valid_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let elements = rows.to_int64() * width.to_int64()
  @utf8.encode(
    (
      $|#include <cuda_bf16.h>
      $|#include <stdint.h>
      $|extern "C" __global__ void \{function_symbol}(const __nv_bfloat16 *normalized, const __nv_bfloat16 *shift, const __nv_bfloat16 *scale, const int32_t *indices, float *output) {
      $|  const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;
      $|  if (i >= \{elements}ULL) return;
      $|  const unsigned long long p = (unsigned long long)indices[i / \{width}ULL] * \{width}ULL + i % \{width}ULL;
      $|  const __nv_bfloat16 s = __float2bfloat16_rn(__fadd_rn(1.0f, __bfloat162float(scale[p])));
      $|  const __nv_bfloat16 v = __float2bfloat16_rn(__fmul_rn(__bfloat162float(normalized[i]), __bfloat162float(s)));
      $|  const __nv_bfloat16 h = __float2bfloat16_rn(__fadd_rn(__bfloat162float(v), __bfloat162float(shift[p])));
      $|  output[i] = __bfloat162float(h);
      $|}
    ),
  )
}
