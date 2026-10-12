///|
// Both directions and compact cache rotation use the same numerical lowering.
// The caller declares left/right/cosine/sine, output and index.
fn rotary_product_source(product : @precision.RotaryProductLaw) -> String {
  match product {
    SeparateFloat32Products =>
      (
        #|    output[index] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(left, cosine), -__fmul_rn(right, sine)));
        #|    output[index + 1ULL] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(left, sine), __fmul_rn(right, cosine)));
      )
    ComplexFloat32Fma =>
      (
        #|    output[index] = __float2bfloat16_rn(__fmaf_rn(left, cosine, -__fmul_rn(right, sine)));
        #|    output[index + 1ULL] = __float2bfloat16_rn(__fmaf_rn(right, cosine, __fmul_rn(left, sine)));
      )
  }
}
