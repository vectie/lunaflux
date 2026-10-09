///|
/// F32 waveform clipping nonlinearity; supports exact input/output aliasing.
pub fn tanh_f32_source(
  elements~ : Int64,
  function_symbol~ : String,
) -> Bytes raise DecoderFoundationCudaSourceError {
  guard elements > 0L && elements <= 2147483647L else { raise InvalidRowCount }
  guard valid_symbol(function_symbol) else { raise InvalidFunctionSymbol }
  @utf8.encode(
    (
      $|#include <math.h>
      $|extern "C" __global__ void \{function_symbol}(const float *input, float *output) {
      $|  const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;
      $|  if (i < \{elements}ULL) output[i] = tanhf(input[i]);
      $|}
    ),
  )
}
