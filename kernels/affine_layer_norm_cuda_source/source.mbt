///|
/// Ordered two-pass F32 mean/centered variance, affine scale and bias, BF16 RNE.
/// A row belongs to one thread in this numerical reference implementation.
pub struct AffineLayerNormCudaSource {
  priv source : Bytes
  priv rows : Int
}

///|
pub(all) suberror AffineLayerNormSourceError {
  InvalidGeometry
  InvalidEpsilon
  InvalidSymbol
} derive(Debug, Eq)

///|
pub extend AffineLayerNormSourceError with Eq::{equal, not_equal}

///|
pub extend AffineLayerNormSourceError with @debug.Debug::{to_repr}

///|
pub fn affine_layer_norm_bf16_source(
  width~ : Int,
  maximum_rows~ : Int,
  epsilon~ : Double,
  function_symbol~ : String,
) -> AffineLayerNormCudaSource raise AffineLayerNormSourceError {
  guard width > 0 &&
    width <= 65536 &&
    maximum_rows > 0 &&
    maximum_rows <= 1048576 else {
    raise InvalidGeometry
  }
  guard epsilon > 0.0 && epsilon <= 1.0 else { raise InvalidEpsilon }
  guard !function_symbol.is_empty() &&
    function_symbol.length() <= 128 &&
    function_symbol[0] is ('a'..='z' | 'A'..='Z' | '_') &&
    function_symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_')) else {
    raise InvalidSymbol
  }
  let source = @utf8.encode(
    (
      $|#include <cuda_bf16.h>
      $|#include <stdint.h>
      $|#include <math.h>
      $|extern "C" __global__ void \{function_symbol}(const int32_t *counts, const __nv_bfloat16 *input, const __nv_bfloat16 *scale, const __nv_bfloat16 *bias, __nv_bfloat16 *output) {
      $|  const int row = (int)blockIdx.x;
      $|  if (threadIdx.x != 0 || counts[3] <= 0 || counts[3] > \{maximum_rows} || row >= counts[3]) return;
      $|  const unsigned long long base = (unsigned long long)row * \{width}ULL;
      $|  float sum = 0.0f;
      $|  for (int c = 0; c < \{width}; ++c) sum = __fadd_rn(sum, __bfloat162float(input[base + c]));
      $|  const float mean = __fdiv_rn(sum, \{width}.0f);
      $|  float variance = 0.0f;
      $|  for (int c = 0; c < \{width}; ++c) { const float d = __fsub_rn(__bfloat162float(input[base + c]), mean); variance = __fadd_rn(variance, __fmul_rn(d, d)); }
      $|  const float inverse = rsqrtf(__fadd_rn(__fdiv_rn(variance, \{width}.0f), \{@luna_cuda_pointwise_aot.canonical_cuda_float_literal(epsilon)}));
      $|  for (int c = 0; c < \{width}; ++c) { const float normalized = __fmul_rn(__fsub_rn(__bfloat162float(input[base + c]), mean), inverse); output[base + c] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(normalized, __bfloat162float(scale[c])), __bfloat162float(bias[c]))); }
      $|}
      $|
    ),
  )
  { source, rows: maximum_rows, }
}

///|
pub fn AffineLayerNormCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn AffineLayerNormCudaSource::launch(self : Self) -> (Int, Int) {
  (self.rows, 1)
}
