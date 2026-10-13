///|
/// Cooperative products, ordered scalar folds, and disjoint control outputs.
/// Three launches implement the physical plan's explicit publication edges.
/// scratch = inverse[rows], mixes[rows, controls]. No runtime allocation.
pub fn ordered_control_tiles_source(
  precision : @precision.HyperConnectionPrecision,
  schedule : @physical.OrderedControlPlan,
  symbol_prefix~ : String,
) -> Bytes raise DecoderHyperConnectionCudaSourceError {
  let (width, hidden, rows) = precision.geometry()
  guard valid_hyper_connection_symbol(symbol_prefix) &&
    width <= 16 &&
    hidden <= 65536 &&
    schedule.geometry() == (rows, width * (width + 2)) &&
    precision.control_square_law() is OrderedFloat32 &&
    precision.control_dot_law() is OrderedFloat32 else {
    raise InvalidNumericalContract
  }
  let flattened = width * hidden
  let controls = width * (width + 2)
  let tile = schedule.product_tile()
  let workers = schedule.workers()
  let norm_eps = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(
    precision.normalization_epsilon(),
  )
  let eps = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(
    precision.control_epsilon(),
  )
  let root = control_root_source(
    "__fadd_rn(__fdiv_rn(sum, \{flattened}.0f), \{norm_eps})",
    precision.control_root_law(),
  )
  let (weight_type, operand, publication) = match precision.control_law() {
    NormalizedBf16Projection =>
      (
        "__nv_bfloat16", "__fmul_rn(__bfloat162float(streams[input_base + column]), inverse)",
        "dot",
      )
    F32ProjectionThenNormalize =>
      (
        "float", "__bfloat162float(streams[input_base + column])", "__fmul_rn(dot, inverse)",
      )
  }
  let weight = if precision.control_law() is NormalizedBf16Projection {
    "__bfloat162float(function_weights[weight_base + column])"
  } else {
    "function_weights[weight_base + column]"
  }
  let sigmoid = control_sigmoid_source(precision.control_sigmoid_law())
  let pre = control_expression(PreOffset, [
    "\{symbol_prefix}_sigmoid(pre_logit)",
    eps,
  ])
  let post = control_expression(PostScale, [
    "\{symbol_prefix}_sigmoid(post_logit)",
  ])
  let pre_logit = control_vector_affine_source(
    "stream",
    "0",
    precision.control_vector_affine_law(),
  )
  let post_logit = control_vector_affine_source(
    "p",
    "1",
    precision.control_vector_affine_law(),
  )
  let affine = matrix_affine_source(
    "control_index",
    precision.control_matrix_affine_law(),
  )
  let exponential = control_expression(SoftmaxWeight, [affine, "maximum"])
  let probability = control_expression(SoftmaxPublish, [
    "initial_combination[index]", "sum", eps,
  ])
  let collapse = control_expression(CollapseUpdate, [
    "sum",
    "pre[stream]",
    "streams[input_base + (unsigned long long)stream * \{hidden}ULL + column]",
  ])
  let publish = control_expression(Publish, ["sum"])
  let matrix_sum = matrix_sum_helper(
    precision.matrix_sum_law(),
    width,
    "\{symbol_prefix}_matrix_sum",
  )
  let sum_update = if precision.matrix_sum_law() is OrderedFloat32 {
    "sum = __fadd_rn(sum, value);"
  } else {
    ""
  }
  let sum_finish = if precision.matrix_sum_law() is ButterflyFloat32 {
    "sum = \{symbol_prefix}_matrix_sum(initial_combination + matrix_base + destination * \{width}, 1);"
  } else {
    ""
  }
  @utf8.encode(
    (
      $|#include <cuda_bf16.h>
      $|#include <math.h>
      $|#include <stdint.h>
      $|__device__ __forceinline__ float \{symbol_prefix}_sigmoid(float x) { \{sigmoid} }
      $|\{matrix_sum}
      $|extern "C" __global__ void \{symbol_prefix}_square_ordered_tiles_v1(const int32_t *counts, const __nv_bfloat16 *streams, float *scratch) {
      $|  const int live_rows = counts[3], row = blockIdx.x, worker = threadIdx.x;
      $|  if (live_rows < 0 || live_rows > \{rows} || row >= live_rows || blockDim.x != \{workers}) return;
      $|  __shared__ float products[\{tile}];
      $|  float sum = 0.0f;
      $|  for (int start = 0; start < \{flattened}; start += \{tile}) {
      $|    const int count = min(\{tile}, \{flattened} - start);
      $|    for (int i = worker; i < count; i += \{workers}) {
      $|      const float x = __bfloat162float(streams[(unsigned long long)row * \{flattened} + start + i]);
      $|      products[i] = isfinite(x) ? __fmul_rn(x, x) : NAN;
      $|    }
      $|    __syncthreads();
      $|    if (worker == 0) for (int i = 0; i < count; ++i) sum = __fadd_rn(sum, products[i]);
      $|    __syncthreads(); // readers retire before the next product tile
      $|  }
      $|  if (worker == 0) scratch[row] = \{root};
      $|}
      $|extern "C" __global__ void \{symbol_prefix}_project_ordered_tiles_v1(const int32_t *counts, const __nv_bfloat16 *streams, const \{weight_type} *function_weights, float *scratch) {
      $|  const int row = blockIdx.x / \{controls}, output = blockIdx.x % \{controls}, worker = threadIdx.x;
      $|  const int live_rows = counts[3]; if (live_rows < 0 || live_rows > \{rows} || row >= live_rows || blockDim.x != \{workers}) return;
      $|  const float inverse = scratch[row]; if (!isfinite(inverse)) return;
      $|  const unsigned long long input_base = (unsigned long long)row * \{flattened};
      $|  const unsigned long long weight_base = (unsigned long long)output * \{flattened};
      $|  __shared__ float products[\{tile}]; float dot = 0.0f;
      $|  for (int start = 0; start < \{flattened}; start += \{tile}) {
      $|    const int count = min(\{tile}, \{flattened} - start);
      $|    for (int i = worker; i < count; i += \{workers}) {
      $|      const int column = start + i;
      $|      products[i] = __fmul_rn(\{operand}, \{weight});
      $|    }
      $|    __syncthreads();
      $|    if (worker == 0) for (int i = 0; i < count; ++i) dot = __fadd_rn(dot, products[i]);
      $|    __syncthreads();
      $|  }
      $|  if (worker == 0) scratch[\{rows} + (unsigned long long)row * \{controls} + output] = \{publication};
      $|}
      $|extern "C" __global__ void \{symbol_prefix}_function_ordered_tiles_v1(const int32_t *counts, const __nv_bfloat16 *streams, const float *scratch, const float *base, const float *scale, float *post, float *initial_combination, __nv_bfloat16 *collapsed) {
      $|  const int live_rows = counts[3], row = blockIdx.x, worker = threadIdx.x;
      $|  if (live_rows < 0 || live_rows > \{rows} || row >= live_rows || blockDim.x != \{workers}) return;
      $|  if (!isfinite(scratch[row])) return;
      $|  const float *mixes = scratch + \{rows} + (unsigned long long)row * \{controls};
      $|  const unsigned long long input_base = (unsigned long long)row * \{flattened};
      $|  __shared__ float pre[\{width}]; __shared__ int valid;
      $|  if (worker == 0) {
      $|    valid = 1;
      $|    for (int i = 0; i < \{controls}; ++i) if (!isfinite(base[i])) valid = 0;
      $|    for (int i = 0; i < 3; ++i) if (!isfinite(scale[i])) valid = 0;
      $|    if (valid) {
      $|      const unsigned long long vector_base = (unsigned long long)row * \{width};
      $|      for (int stream = 0; stream < \{width}; ++stream) {
      $|        const float pre_logit = \{pre_logit}; pre[stream] = \{pre};
      $|        const int p = \{width} + stream; const float post_logit = \{post_logit}; post[vector_base + stream] = \{post};
      $|      }
      $|      const unsigned long long matrix_base = (unsigned long long)row * \{width * width};
      $|      for (int destination = 0; destination < \{width}; ++destination) {
      $|        int control_index = 2 * \{width} + destination * \{width}; float maximum = \{affine};
      $|        for (int source = 1; source < \{width}; ++source) { control_index = 2 * \{width} + destination * \{width} + source; const float value = \{affine}; if (value > maximum) maximum = value; }
      $|        float sum = 0.0f;
      $|        for (int source = 0; source < \{width}; ++source) {
      $|          control_index = 2 * \{width} + destination * \{width} + source;
      $|          const float value = \{exponential}; const unsigned long long index = matrix_base + destination * \{width} + source;
      $|          initial_combination[index] = value; \{sum_update}
      $|        }
      $|        \{sum_finish}
      $|        for (int source = 0; source < \{width}; ++source) { const unsigned long long index = matrix_base + destination * \{width} + source; initial_combination[index] = \{probability}; }
      $|      }
      $|    }
      $|  }
      $|  __syncthreads(); if (!valid) return;
      $|  for (int column = worker; column < \{hidden}; column += \{workers}) {
      $|    float sum = 0.0f;
      $|    for (int stream = 0; stream < \{width}; ++stream) sum = \{collapse};
      $|    if (isfinite(sum)) collapsed[(unsigned long long)row * \{hidden} + column] = \{publish};
      $|  }
      $|}
      $|
    ),
  )
}
