///|
async test "flagship binds centered gated norms, full routing and grouped recurrent state" {
  let model = @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
  let operators = @minicpm_v47_operators.DecoderOperators::new(
    model,
    maximum_rows=32,
  )
  let query = operators.normalization(FullQuery)
  assert_eq(query.geometry(), (256, 16, 32))
  assert_eq(query.input_layout(), (512, 8192L))
  assert_true(query.mode() is Weighted(OnePlusGamma))
  assert_eq(operators.normalization(FullKey).geometry(), (256, 2, 32))
  assert_eq(operators.normalization(Decoder).geometry(), (2048, 1, 32))
  assert_eq(operators.normalization(RecurrentOutput).geometry(), (128, 32, 32))
  assert_true(
    operators.normalization(RecurrentOutput).mode()
    is Bf16NormBf16WeightSiluGate,
  )
  assert_eq(operators.routing().geometry(), (256, 8, 32))
  assert_true(operators.routing().weight_storage() is Float32CarryingBf16)
  assert_eq(operators.routed_experts().geometry(), (32, 2048, 512, 256, 8))
  assert_true(operators.routed_experts().combine_rounding() is Bf16EachAdd)
  assert_eq(operators.shared_mlp().geometry(), (2048, 512, 32))
  assert_eq(operators.shared_combine().geometry(), (2048, 32))
  let rotary = operators.query_rotary()
  assert_eq(rotary.geometry(), (16, 256, 64, 32, 262144))
  assert_eq(operators.key_rotary().geometry(), (2, 256, 64, 32, 262144))
  assert_eq(rotary.axes().iter().to_array(), Array::makei(32, i => i % 3))
  assert_eq(rotary.axes().filter(axis => axis == 0).length(), 11)
  assert_eq(rotary.axes().filter(axis => axis == 1).length(), 11)
  assert_eq(rotary.axes().filter(axis => axis == 2).length(), 10)
  assert_eq(rotary.inverse_frequencies()[0], 1.0F)
  assert_eq(operators.full_output_gate().geometry(), (256, 16, 32))
  assert_eq(operators.full_output_gate().gate_layout(), (512, 8192L, 256))
  assert_eq(
    operators.full_projection(QueryAndGate).geometry(),
    (32, 8192, 2048),
  )
  assert_eq(operators.full_projection(Key).geometry(), (32, 512, 2048))
  assert_eq(operators.full_projection(Value).geometry(), (32, 512, 2048))
  assert_eq(operators.full_projection(Output).geometry(), (32, 2048, 4096))
  assert_eq(operators.controls().geometry(), (32, 32))
  assert_eq(operators.convolution().state_bytes(), 49152L)
  assert_true(operators.convolution().projection_boundary() is Bf16BeforeSilu)
  assert_true(
    operators.convolution().activation_expression().root()
    is Convert(
      BFloat16,
      Binary(DivideRN, Convert(Float32, Convert(BFloat16, Input(0))), _)
    ),
  )
  assert_eq(operators.recurrent().query_key_head_count(), 16)
  assert_true(operators.recurrent().decay_layout() is PerValueHead)
}
