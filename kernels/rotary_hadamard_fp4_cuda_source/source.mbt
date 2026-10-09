///|
pub struct RotaryHadamardFp4CudaSource {
  priv plan : @precision.RotaryHadamardFp4Precision
  priv source : Bytes
  priv symbols : ReadOnlyArray[String]
}

///|
pub fn rotary_hadamard_fp4_source(
  plan : @precision.RotaryHadamardFp4Precision,
  symbol_prefix~ : String,
) -> RotaryHadamardFp4CudaSource raise {
  let symbols = [
    symbol_prefix + "_rotary",
    symbol_prefix + "_hadamard",
    symbol_prefix + "_simulate",
  ]
  // The shared renderer also validates the symbol prefix.
  let rotary = @rotary.rotary_tensor_source(plan.rotary(), symbol=symbols[0])
  let (heads, width, _, rows, _) = plan.rotary().geometry()
  let block = plan.block_width()
  let normalization = @literal.canonical_cuda_float_literal(
    plan.normalization(),
  )
  guard block <= 1024 && heads * (width / block) <= 65535 else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let source = @utf8.encode(
    (
      $|#include <cuda_bf16.h>
      $|#include <math.h>
      $|#include <stdint.h>
      $|\{rotary}
      $|extern "C" __global__ void \{symbols[1]}(const int32_t *counts, const __nv_bfloat16 *input, __nv_bfloat16 *output) {
      $|  const int row = (int)blockIdx.x, head = (int)blockIdx.y, live = counts[3];
      $|  if (live <= 0 || live > \{rows} || row >= live || head >= \{heads}) return;
      $|  const unsigned long long base = ((unsigned long long)row * \{heads}ULL + head) * \{width}ULL;
      $|  __shared__ float first[\{width}], second[\{width}]; float *old_values = first, *new_values = second;
      $|  for (int col = threadIdx.x; col < \{width}; col += blockDim.x) old_values[col] = __bfloat162float(input[base + col]);
      $|  __syncthreads();
      $|  for (int stride = 1; stride < \{width}; stride <<= 1) {
      $|    for (int col = threadIdx.x; col < \{width}; col += blockDim.x) {
      $|      const float left = old_values[col & ~stride], right = old_values[col | stride];
      $|      new_values[col] = __fadd_rn(left, (col & stride) ? -right : right);
      $|    }
      $|    __syncthreads(); float *swap = old_values; old_values = new_values; new_values = swap;
      $|  }
      $|  const float normalization = \{normalization};
      $|  for (int col = threadIdx.x; col < \{width}; col += blockDim.x) output[base + col] = __float2bfloat16_rn(__fmul_rn(old_values[col], normalization));
      $|}
      $|extern "C" __global__ void \{symbols[2]}(const int32_t *counts, __nv_bfloat16 *values) {
      $|  const int row = (int)blockIdx.x, group = (int)blockIdx.y, live = counts[3], col = threadIdx.x;
      $|  if (live <= 0 || live > \{rows} || row >= live || group >= \{heads * (width / block)}) return;
      $|  const unsigned long long base = ((unsigned long long)row * \{heads * width}ULL) + group * \{block}ULL;
      $|  const float value = __bfloat162float(values[base + col]); __shared__ float maximum[\{block}];
      $|  maximum[col] = fmaxf(fabsf(value), 0x1.8p-124f); __syncthreads();
      $|  for (int stride = \{block / 2}; stride > 0; stride >>= 1) { if (col < stride) maximum[col] = fmaxf(maximum[col], maximum[col + stride]); __syncthreads(); }
      $|  const unsigned bits = __float_as_uint(__fmul_rn(maximum[0], 0.16666666666666667f));
      $|  const unsigned exponent = (bits >> 23) & 0xffU;
      $|  const float scale = __uint_as_float((exponent + ((bits & 0x7fffffU) != 0U ? 1U : 0U)) << 23);
      $|  const float normalized = fminf(6.0f, fabsf(__fdiv_rn(value, scale)));
      $|  const float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f}; int selected = 0; float distance = normalized;
      $|  for (int code = 1; code < 8; ++code) { const float error = fabsf(normalized - levels[code]); if (error < distance || (error == distance && !(code & 1))) { selected = code; distance = error; } }
      $|  values[base + col] = __float2bfloat16_rn(copysignf(__fmul_rn(levels[selected], scale), value));
      $|}
    ),
  )
  { plan, source, symbols: ReadOnlyArray::from_array(symbols), }
}

///|
pub fn RotaryHadamardFp4CudaSource::plan(
  self : Self,
) -> @precision.RotaryHadamardFp4Precision {
  self.plan
}

///|
pub fn RotaryHadamardFp4CudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn RotaryHadamardFp4CudaSource::symbol(self : Self, stage : Int) -> String {
  self.symbols[stage]
}

///|
pub fn RotaryHadamardFp4CudaSource::launch(
  self : Self,
  stage : Int,
) -> (Int, Int, Int) {
  let (heads, width, _, rows, _) = self.plan.rotary().geometry()
  if stage == 2 {
    (rows, heads * (width / self.plan.block_width()), self.plan.block_width())
  } else {
    (rows, heads, 256)
  }
}
