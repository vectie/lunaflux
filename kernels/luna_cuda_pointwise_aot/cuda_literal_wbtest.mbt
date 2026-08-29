///|
test "canonical CUDA float literals preserve integer decimal and exponent syntax" {
  assert_eq(canonical_cuda_float_literal(10000.0), "10000.0f")
  assert_eq(canonical_cuda_float_literal(0.00001), "0.00001f")
  assert_eq(canonical_cuda_float_literal(1.25), "1.25f")
}
