///|
test "normalization preserves model-specific gamma and intermediate BF16 rounds" {
  let centered = @precision_ir.RmsNormPrecision::new(
    vector_width=256,
    vectors_per_row=16,
    maximum_rows=32,
    epsilon=0.000001,
    mode=Weighted(OnePlusGamma),
    input_vector_stride=512,
    input_row_stride=8192L,
  )
  assert_eq(centered.input_layout(), (512, 8192L))
  assert_eq(centered.output_bytes(), 262144L)
  assert_eq(centered.square_fold().extent, 256)
  assert_true(
    centered.expression(Weight).root()
    is Binary(
      MultiplyRN,
      Input(0),
      Binary(AddRN, Float32Bits(0x3f800000U), Convert(Float32, Input(1)))
    ),
  )
  let gated = @precision_ir.RmsNormPrecision::new(
    vector_width=128,
    vectors_per_row=32,
    maximum_rows=32,
    epsilon=0.000001,
    mode=Bf16NormBf16WeightSiluGate,
  )
  assert_true(
    gated.expression(Normalize).root() is Convert(Float32, Convert(BFloat16, _)),
  )
  assert_true(
    gated.expression(Weight).root() is Convert(Float32, Convert(BFloat16, _)),
  )
  assert_true(gated.expression(Gate).root() is Convert(BFloat16, _))
}
