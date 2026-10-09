///|
/// Bit-preserving two-byte element copy. Source and destination must not
/// overlap; no arithmetic or BF16 canonicalization is performed.
pub fn copy_bf16_source(
  elements~ : Int64,
  function_symbol~ : String,
) -> Bytes raise DecoderFoundationCudaSourceError {
  guard elements > 0L && elements <= 2147483647L else { raise InvalidRowCount }
  guard valid_symbol(function_symbol) else { raise InvalidFunctionSymbol }
  @utf8.encode(
    (
      $|extern "C" __global__ void \{function_symbol}(const unsigned short *input, unsigned short *output) {
      $|  const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;
      $|  if (i < \{elements}ULL) output[i] = input[i];
      $|}
    ),
  )
}
