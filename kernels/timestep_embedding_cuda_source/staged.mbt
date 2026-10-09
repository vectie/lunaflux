///|
/// Materialize the first dense+SiLU once per hidden element, rather than
/// recomputing it for every output column. Both stages preserve the existing
/// bias-first ordered F32 arithmetic contract and use 256 threads per CTA.
pub fn staged_dense_silu_dense_f32_rows_source(
  rows~ : Int,
  input_width~ : Int,
  hidden_width~ : Int,
  output_width~ : Int,
  function_prefix~ : String,
) -> Bytes raise {
  guard rows > 0 else {
    raise TimestepEmbeddingCudaSourceError::InvalidRowCount
  }
  guard input_width > 0 &&
    hidden_width > 0 &&
    output_width > 0 &&
    rows.to_int64() <= 2147483647L / hidden_width.to_int64() &&
    rows.to_int64() <= 2147483647L / output_width.to_int64() else {
    raise TimestepEmbeddingCudaSourceError::InvalidFrequencyWidth
  }
  guard valid_symbol(function_prefix + "_hidden") else {
    raise TimestepEmbeddingCudaSourceError::InvalidFunctionSymbol
  }
  let elements = rows.to_int64() * hidden_width.to_int64()
  let first = @utf8.encode(
    (
      $|#include <math.h>
      $|extern "C" __global__ void \{function_prefix}_hidden(const float *input, const float *weight, const float *bias, float *output) {
      $|  const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;
      $|  if (i >= \{elements}ULL) return;
      $|  const unsigned long long row = i / \{hidden_width}ULL, hidden = i % \{hidden_width}ULL;
      $|  float first = bias[hidden];
      $|  for (int c = 0; c < \{input_width}; ++c)
      $|    first = __fadd_rn(first, __fmul_rn(input[row * \{input_width}ULL + c], weight[hidden * \{input_width}ULL + c]));
      $|  output[i] = __fdiv_rn(first, __fadd_rn(1.0f, expf(-first)));
      $|}
    ),
  )
  first +
  @decoder_projection_cuda_source.dense_projection_f32_bias_f32_source(
    rows=rows.to_int64(),
    input_width=hidden_width,
    output_width~,
    function_symbol=function_prefix + "_output",
  ).source_bytes()
}
