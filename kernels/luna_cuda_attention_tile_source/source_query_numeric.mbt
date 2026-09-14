///|
/// If-convert a pure scalar selection without changing expf or its rounding.
/// CUDA has no observable floating-point trap/errno here: evaluating the
/// unselected exponential is safe, including inf-inf. A bit mask (not a
/// multiply) maps that result to positive zero, even when it is NaN.
/// BF16 pair packing preserves the two independent round-to-nearest results.
fn cuda_query_owned_numeric_primitives() -> String {
  (
    #|static __device__ __forceinline__ float lf_masked_exp(float score, float delta) {
    #|  const float value=expf(delta);
    #|  const unsigned mask=0U-(unsigned)!isinf(score);
    #|  return __uint_as_float(__float_as_uint(value)&mask);
    #|}
    #|static __device__ __forceinline__ unsigned lf_probability_pair(float low, float high) {
    #|  unsigned packed;
    #|  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(packed) : "f"(high), "f"(low));
    #|  return packed;
    #|}
  )
}
