///|
/// One shared lowering for query/KV and standalone compressed-tensor paths.
/// It modifies only the prefix and preserves the rotated suffix's BF16 bits.
fn cache_simulation_source(
  symbol : String,
  rows : Int,
  heads : Int,
  width : Int,
  prefix : Int,
  block : Int,
  floor : Double,
  scale : @precision.SimulatedCacheScale,
) -> String raise {
  guard heads * (prefix / block) <= 65535 else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let floor_f = @literal.canonical_cuda_float_literal(floor)
  let scale_code = match scale {
    LinearF32 => "const float scale = requested;"
    PowerOfTwo =>
      "const unsigned bits = __float_as_uint(requested); const unsigned exponent = (bits >> 23) & 0xffU; const float scale = __uint_as_float((exponent + ((bits & 0x7fffffU) != 0U ? 1U : 0U)) << 23);"
  }
  (
    $|extern "C" __global__ void \{symbol}(const int32_t *counts, __nv_bfloat16 *kv) {
    $|  const int row = (int)blockIdx.x, group = (int)blockIdx.y, live = counts[3];
    $|  if (live <= 0 || live > \{rows} || row >= live || group >= \{heads * (prefix / block)}) return;
    $|  const int head = group / \{if prefix == 0 { 1 } else { prefix / block }}, block = group % \{if prefix == 0 { 1 } else { prefix / block }};
    $|  const unsigned long long base = ((unsigned long long)row * \{heads}ULL + head) * \{width}ULL + block * \{block}ULL;
    $|  __shared__ float maxima[\{block}]; const int col = threadIdx.x;
    $|  const float value = __bfloat162float(kv[base + col]); maxima[col] = fmaxf(\{floor_f}, fabsf(value)); __syncthreads();
    $|  for (int stride = \{block / 2}; stride > 0; stride >>= 1) { if (col < stride) maxima[col] = fmaxf(maxima[col], maxima[col + stride]); __syncthreads(); }
    $|  const float requested = __fmul_rn(maxima[0], 0.002232142857142857f); \{scale_code}
    $|  const float normalized = fminf(448.0f, fmaxf(-448.0f, __fdiv_rn(value, scale)));
    $|  __nv_fp8_e4m3 rounded; rounded.__x = __nv_cvt_float_to_fp8(normalized, __NV_SATFINITE, __NV_E4M3);
    $|  kv[base + col] = __float2bfloat16_rn(__fmul_rn((float)rounded, scale));
    $|}
  )
}
