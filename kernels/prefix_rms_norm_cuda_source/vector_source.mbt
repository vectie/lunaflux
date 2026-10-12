///|
/// Device geometry implements the semantic fold; its launch requirement travels
/// with the lowered source instead of being rediscovered by the runtime.
fn vector_prefix_norm_source(
  prefix_width~ : Int,
  suffix_width~ : Int,
  maximum_rows~ : Int,
  epsilon~ : Double,
  function_symbol~ : String,
  square_law~ : @precision.SquareReductionLaw,
) -> PrefixRmsNormCudaSource raise PrefixRmsNormCudaSourceError {
  guard square_law is VectorizedTreeFloat32(partitions~, vector_width~) &&
    partitions > 0 &&
    partitions <= 1024 &&
    (partitions & (partitions - 1)) == 0 &&
    vector_width > 0 &&
    vector_width <= 4 &&
    (vector_width & (vector_width - 1)) == 0 else {
    raise UnsupportedSquareReductionLaw
  }
  let suffix_parameter = if suffix_width > 0 {
    ", __nv_bfloat16 *suffix_output"
  } else {
    ""
  }
  let suffix_copy = if suffix_width > 0 {
    "for (int column = threadIdx.x; column < \{suffix_width}; column += \{partitions}) suffix_output[(unsigned long long)row * \{suffix_width}ULL + column] = input[input_base + \{prefix_width}ULL + column];"
  } else {
    ""
  }
  let eps = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(epsilon)
  let text =
    $|#include <cuda_bf16.h>
    $|#include <stdint.h>
    $|#include <math.h>
    $|extern "C" __global__ void \{function_symbol}(const int32_t *counts, const __nv_bfloat16 *input, const __nv_bfloat16 *weight, __nv_bfloat16 *normalized_output\{suffix_parameter}) {
    $|  const int row = blockIdx.x, lane = threadIdx.x;
    $|  const int row_count = counts[3];
    $|  if (row_count <= 0 || row_count > \{maximum_rows} || row >= row_count || blockDim.x != \{partitions}) return;
    $|  __shared__ float partial[\{partitions}]; float sums[\{vector_width}] = {};
    $|  const unsigned long long input_base = (unsigned long long)row * \{prefix_width + suffix_width}ULL;
    $|  for (int base = lane * \{vector_width}; base < \{prefix_width}; base += \{partitions * vector_width}) {
    $|    for (int component = 0; component < \{vector_width} && base + component < \{prefix_width}; ++component) {
    $|      const float value = __bfloat162float(input[input_base + base + component]);
    $|      sums[component] = __fadd_rn(sums[component], __fmul_rn(value, value));
    $|    }
    $|  }
    $|  for (int component = 1; component < \{vector_width}; ++component) sums[0] = __fadd_rn(sums[0], sums[component]);
    $|  partial[lane] = sums[0]; __syncthreads();
    $|  for (int span = \{partitions / 2}; span > 0; span >>= 1) { if (lane < span) partial[lane] = __fadd_rn(partial[lane], partial[lane + span]); __syncthreads(); }
    $|  const float inverse = rsqrtf(__fadd_rn(__fdiv_rn(partial[0], \{prefix_width}.0f), \{eps}));
    $|  const unsigned long long output_base = (unsigned long long)row * \{prefix_width}ULL;
    $|  for (int column = lane; column < \{prefix_width}; column += \{partitions}) normalized_output[output_base + column] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(input[input_base + column]), inverse), __bfloat162float(weight[column])));
    $|  \{suffix_copy}
    $|}
  {
    operation: PrefixRmsNormBf16WithOptionalSuffixCopy,
    numerical_contract: VectorizedTreeFloat32PrefixReductionBf16RoundAndBitExactSuffixCopyV1(
      square_law,
    ),
    prefix_width,
    suffix_width,
    maximum_rows,
    epsilon,
    function_symbol,
    source: @utf8.encode(text),
    threads: partitions,
  }
}
