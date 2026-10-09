///|
/// Ungated BF16 residual addition. Exact in-place output aliasing either input
/// is permitted; each thread reads both elements before its single store.
pub fn residual_add_bf16_source(
  elements~ : Int64,
  function_symbol~ : String,
) -> Bytes raise DecoderFoundationCudaSourceError {
  guard elements > 0L && elements <= 2147483647L else { raise InvalidRowCount }
  guard valid_symbol(function_symbol) else { raise InvalidFunctionSymbol }
  @utf8.encode(
    prelude() +
    (
      $|extern "C" __global__ void \{function_symbol}(const __nv_bfloat16 *state, const __nv_bfloat16 *branch, __nv_bfloat16 *output) {
      $|  const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;
      $|  if (i >= \{elements}ULL) return;
      $|  output[i] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(state[i]), __bfloat162float(branch[i])));
      $|}
    ),
  )
}
