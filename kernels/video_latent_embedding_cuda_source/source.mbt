///|
pub(all) suberror EmbeddingError {
  InvalidGeometry
  InvalidSymbol
} derive(Eq)

///|
pub struct EmbeddingSource {
  source : Bytes
  symbol : String
  input_elements : Int64
  output_elements : Int64
}

///|
/// F32 channel-major volume to row-major projection, followed by learned
/// register rows and one zero class row. This device lowering is independent
/// of model family. Reductions use ordered separately rounded F32 operations.
pub fn render(
  batch~ : Int,
  positions~ : Int,
  input_channels~ : Int,
  hidden_channels~ : Int,
  register_rows~ : Int,
  symbol~ : String,
) -> EmbeddingSource raise EmbeddingError {
  guard batch > 0 &&
    positions > 0 &&
    input_channels > 0 &&
    hidden_channels > 0 &&
    register_rows >= 0 &&
    register_rows < 2147483647 - positions else {
    raise InvalidGeometry
  }
  let rows = positions + register_rows + 1
  guard batch.to_int64() * rows.to_int64() <=
    2147483647L / hidden_channels.to_int64() &&
    batch.to_int64() * positions.to_int64() <=
    2147483647L / input_channels.to_int64() else {
    raise InvalidGeometry
  }
  guard !symbol.is_empty() &&
    symbol.length() <= 128 &&
    symbol[0] is ('a'..='z' | 'A'..='Z' | '_') &&
    symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_')) else {
    raise InvalidSymbol
  }
  let output_elements = batch.to_int64() *
    rows.to_int64() *
    hidden_channels.to_int64()
  let source = @utf8.encode(
    (
      $|#include <stdint.h>
      $|extern "C" __global__ void \{symbol}(const float *input, const float *weight, const float *bias, const float *registers, float *output) {
      $|  const unsigned long long flat = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;
      $|  if (flat >= \{output_elements}ULL) return;
      $|  const unsigned long long channel = flat % \{hidden_channels}ULL;
      $|  const unsigned long long row = (flat / \{hidden_channels}ULL) % \{rows}ULL;
      $|  const unsigned long long batch_index = flat / (\{hidden_channels}ULL * \{rows}ULL);
      $|  if (row >= \{positions}ULL) {
      $|    output[flat] = row < \{positions + register_rows}ULL ? registers[(row - \{positions}ULL) * \{hidden_channels}ULL + channel] : 0.0f;
      $|    return;
      $|  }
      $|  float sum = bias[channel];
      $|  for (int k = 0; k < \{input_channels}; ++k) {
      $|    const float product = __fmul_rn(input[(batch_index * \{input_channels}ULL + k) * \{positions}ULL + row], weight[channel * \{input_channels}ULL + k]);
      $|    sum = __fadd_rn(sum, product);
      $|  }
      $|  output[flat] = sum;
      $|}
    ),
  )
  {
    source,
    symbol,
    input_elements: batch.to_int64() *
    positions.to_int64() *
    input_channels.to_int64(),
    output_elements,
  }
}
