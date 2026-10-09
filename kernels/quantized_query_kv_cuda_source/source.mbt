///|
/// Six independent semantic stages, prepared once and borrowed by a decoder
/// queue. FP8 projections share versioned helpers, not model-specific kernels.
pub struct QuantizedQueryKvCudaSource {
  priv plan : @precision.QuantizedQueryKvPrecision
  priv source : Bytes
  priv symbols : ReadOnlyArray[String]
  priv geometry : ReadOnlyArray[(Int, Int, Int)]
}

///|
pub fn quantized_query_kv_source(
  plan : @precision.QuantizedQueryKvPrecision,
  symbol_prefix~ : String,
) -> QuantizedQueryKvCudaSource raise {
  guard plan.activation_quantization().amax_floor() == 0.0001 &&
    plan.head_normalization() is Bf16SquareMeanEpsilonInverseProduct else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let (hidden, rank, heads, width, rows) = plan.geometry()
  let symbols = ["q_a", "q_norm", "q_b", "q_head_norm", "kv", "kv_norm"].map(name => {
    symbol_prefix + "_" + name
  })
  let sources = Array::make(6, b"")
  let geometry = Array::make(6, (rows, 1, 256))
  for
    (input, output, stage) in [
      (hidden, rank, 0),
      (rank, heads * width, 2),
      (hidden, width, 4),
    ] {
    sources[stage] = @projection.block_fp8_ue8m0_projection_cuda_source(
      input_width=input,
      output_width=output,
      maximum_rows=rows,
      function_symbol=symbols[stage],
    ).source_bytes()
    geometry[stage] = (rows, (output + 255) / 256, 256)
  }
  for (size, stage) in [(rank, 1), (width, 5)] {
    sources[stage] = @norm.prefix_rms_norm_bf16_with_optional_suffix_copy_source(
      prefix_width=size,
      suffix_width=0,
      maximum_rows=rows,
      epsilon=plan.normalization_epsilon(),
      function_symbol=symbols[stage],
    ).source_bytes()
  }
  let epsilon = @literal.canonical_cuda_float_literal(
    plan.normalization_epsilon(),
  )
  sources[3] = @utf8.encode(
    (
      $|#include <cuda_bf16.h>
      $|#include <math.h>
      $|extern "C" __global__ void \{symbols[3]}(const int *counts, const __nv_bfloat16 *input, __nv_bfloat16 *output) {
      $|  const int row = (int)blockIdx.x; const int head = (int)blockIdx.y;
      $|  const int live = counts[3]; if (live <= 0 || live > \{rows} || row >= live || head >= \{heads}) return;
      $|  const unsigned long long base = ((unsigned long long)row * \{heads}ULL + head) * \{width}ULL;
      $|  __shared__ float partial[256]; float sum = 0.0f;
      $|  for (int col = threadIdx.x; col < \{width}; col += 256) {
      $|    const float value = __bfloat162float(input[base + col]);
      $|    const float square = __bfloat162float(__float2bfloat16_rn(__fmul_rn(value, value)));
      $|    sum = __fadd_rn(sum, square);
      $|  }
      $|  partial[threadIdx.x] = sum; __syncthreads();
      $|  for (int stride = 128; stride > 0; stride >>= 1) {
      $|    if (threadIdx.x < stride) partial[threadIdx.x] = __fadd_rn(partial[threadIdx.x], partial[threadIdx.x + stride]);
      $|    __syncthreads();
      $|  }
      $|  const float mean = __bfloat162float(__float2bfloat16_rn(__fdiv_rn(partial[0], \{width}.0f)));
      $|  const float variance = __bfloat162float(__float2bfloat16_rn(__fadd_rn(mean, \{epsilon})));
      $|  const float inverse = __bfloat162float(__float2bfloat16_rn(rsqrtf(variance)));
      $|  for (int col = threadIdx.x; col < \{width}; col += 256)
      $|    output[base + col] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(input[base + col]), inverse));
      $|}
      $|
    ),
  )
  geometry[3] = (rows, heads, 256)
  {
    plan,
    source: sources.fold(init=b"", (all, part) => all + part + b"\n"),
    symbols: ReadOnlyArray::from_array(symbols),
    geometry: ReadOnlyArray::from_array(geometry),
  }
}

///|
pub fn QuantizedQueryKvCudaSource::plan(
  self : Self,
) -> @precision.QuantizedQueryKvPrecision {
  self.plan
}

///|
pub fn QuantizedQueryKvCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn QuantizedQueryKvCudaSource::symbol(self : Self, stage : Int) -> String {
  self.symbols[stage]
}

///|
/// Grid X, grid Y, block X; head and output-column axes remain separate.
pub fn QuantizedQueryKvCudaSource::launch(
  self : Self,
  stage : Int,
) -> (Int, Int, Int) {
  self.geometry[stage]
}
