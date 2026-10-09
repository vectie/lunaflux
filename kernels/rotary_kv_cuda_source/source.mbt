///|
/// Physical layout is `[row,head,full_width]` for both Q and shared KV.
/// The two functions are borrowed by the surrounding ordered executor.
pub struct RotaryKvCudaSource {
  priv plan : @precision.RotaryKvPrecision
  priv source : Bytes
  priv symbols : ReadOnlyArray[String]
}

///|
fn valid_symbol(symbol : StringView) -> Bool {
  !symbol.is_empty() &&
  symbol.length() <= 112 &&
  symbol[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
pub fn rotary_kv_source(
  plan : @precision.RotaryKvPrecision,
  symbol_prefix~ : String,
) -> RotaryKvCudaSource raise {
  guard valid_symbol(symbol_prefix) else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let (qh, kh, width, rotary, rows, positions) = plan.geometry()
  let prefix = width - rotary
  let symbols = [symbol_prefix + "_rotary", symbol_prefix + "_simulate"]
  let frequency = match plan.frequency() {
    Base(theta~) =>
      "const float frequency = __fdiv_rn(1.0f, powf(" +
      @literal.canonical_cuda_float_literal(theta) +
      ", __fdiv_rn((float)(pair * 2), \{rotary}.0f)));"
    Yarn(theta~, original_positions~, factor~, beta_fast~, beta_slow~) => {
      let theta_f = @literal.canonical_cuda_float_literal(theta)
      let factor_f = @literal.canonical_cuda_float_literal(factor)
      (
        $|const double fast = ((double)\{rotary} * log((double)\{original_positions} / ((double)\{beta_fast} * 6.2831853071795864769))) / (2.0 * log((double)\{theta}));
        $|const double slow = ((double)\{rotary} * log((double)\{original_positions} / ((double)\{beta_slow} * 6.2831853071795864769))) / (2.0 * log((double)\{theta}));
        $|const float low = (float)fmax(0.0, floor(fast)); const float high = (float)fmin(\{rotary - 1}.0, ceil(slow));
        $|const float span = __fadd_rn(high, __fadd_rn(-low, high == low ? 0.001f : 0.0f));
        $|const float ramp = fminf(1.0f, fmaxf(0.0f, __fdiv_rn(__fadd_rn((float)pair, -low), span)));
        $|const float smooth = __fadd_rn(1.0f, -ramp);
        $|const float base_frequency = __fdiv_rn(1.0f, powf(\{theta_f}, __fdiv_rn((float)(pair * 2), \{rotary}.0f)));
        $|const float frequency = __fadd_rn(__fmul_rn(__fdiv_rn(base_frequency, \{factor_f}), __fadd_rn(1.0f, -smooth)), __fmul_rn(base_frequency, smooth));
      )
    }
  }
  let (block, floor, scale) = plan.simulation()
  guard kh * (prefix / block) <= 65535 else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let floor_f = @literal.canonical_cuda_float_literal(floor)
  let scale_code = match scale {
    LinearF32 => "const float scale = requested;"
    PowerOfTwo =>
      "const unsigned bits = __float_as_uint(requested); const unsigned exponent = (bits >> 23) & 0xffU; const float scale = __uint_as_float((exponent + ((bits & 0x7fffffU) != 0U ? 1U : 0U)) << 23);"
  }
  let source = @utf8.encode(
    (
      $|#include <cuda.h>
      $|#include <cuda_bf16.h>
      $|#include <cuda_fp8.h>
      $|#include <math.h>
      $|#include <stdint.h>
      $|extern "C" __global__ void \{symbols[0]}(const int32_t *counts, const int32_t *positions, const uint16_t *query, const uint16_t *kv, uint16_t *query_out, uint16_t *kv_out) {
      $|  const int row = (int)blockIdx.x, head = (int)blockIdx.y, live = counts[3];
      $|  if (live <= 0 || live > \{rows} || row >= live || head >= \{qh + kh}) return;
      $|  const int position = positions[row]; if (position < 0 || position >= \{positions}) return;
      $|  const bool is_query = head < \{qh}; const int heads = is_query ? \{qh} : \{kh}; const int local_head = is_query ? head : head - \{qh};
      $|  const unsigned long long base = ((unsigned long long)row * heads + local_head) * \{width}ULL;
      $|  const uint16_t *input_words = is_query ? query : kv; uint16_t *output_words = is_query ? query_out : kv_out;
      $|  for (int col = threadIdx.x; col < \{prefix}; col += blockDim.x) output_words[base + col] = input_words[base + col];
      $|  for (int pair = threadIdx.x; pair < \{rotary / 2}; pair += blockDim.x) {
      $|    \{frequency}
      $|    const float angle = __fmul_rn((float)position, frequency), cosine = cosf(angle), sine = sinf(angle);
      $|    const unsigned long long index = base + \{prefix}ULL + pair * 2ULL;
      $|    const __nv_bfloat16 *input = (const __nv_bfloat16 *)input_words; __nv_bfloat16 *output = (__nv_bfloat16 *)output_words;
      $|    const float left = __bfloat162float(input[index]), right = __bfloat162float(input[index + 1ULL]);
      $|    output[index] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(left, cosine), -__fmul_rn(right, sine)));
      $|    output[index + 1ULL] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(left, sine), __fmul_rn(right, cosine)));
      $|  }
      $|}
      $|extern "C" __global__ void \{symbols[1]}(const int32_t *counts, __nv_bfloat16 *kv) {
      $|  const int row = (int)blockIdx.x, group = (int)blockIdx.y, live = counts[3];
      $|  if (live <= 0 || live > \{rows} || row >= live || group >= \{kh * (prefix / block)}) return;
      $|  const int head = group / \{if prefix == 0 { 1 } else { prefix / block }}, block = group % \{if prefix == 0 { 1 } else { prefix / block }};
      $|  const unsigned long long base = ((unsigned long long)row * \{kh}ULL + head) * \{width}ULL + block * \{block}ULL;
      $|  __shared__ float maxima[\{block}]; const int col = threadIdx.x;
      $|  const float value = __bfloat162float(kv[base + col]); maxima[col] = fmaxf(\{floor_f}, fabsf(value)); __syncthreads();
      $|  for (int stride = \{block / 2}; stride > 0; stride >>= 1) { if (col < stride) maxima[col] = fmaxf(maxima[col], maxima[col + stride]); __syncthreads(); }
      $|  const float requested = __fmul_rn(maxima[0], 0.002232142857142857f); \{scale_code}
      $|  const float normalized = fminf(448.0f, fmaxf(-448.0f, __fdiv_rn(value, scale)));
      $|  __nv_fp8_e4m3 rounded; rounded.__x = __nv_cvt_float_to_fp8(normalized, __NV_SATFINITE, __NV_E4M3);
      $|  kv[base + col] = __float2bfloat16_rn(__fmul_rn((float)rounded, scale));
      $|}
      $|
    ),
  )
  { plan, source, symbols: ReadOnlyArray::from_array(symbols), }
}

///|
pub fn RotaryKvCudaSource::plan(self : Self) -> @precision.RotaryKvPrecision {
  self.plan
}

///|
pub fn RotaryKvCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn RotaryKvCudaSource::symbol(self : Self, stage : Int) -> String {
  self.symbols[stage]
}

///|
pub fn RotaryKvCudaSource::launch(self : Self, stage : Int) -> (Int, Int, Int) {
  let (qh, kh, width, rotary, rows, _) = self.plan.geometry()
  let (block, _, _) = self.plan.simulation()
  if stage == 0 {
    (rows, qh + kh, 256)
  } else {
    (rows, Int::max(1, kh * ((width - rotary) / block)), block)
  }
}
