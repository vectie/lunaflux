///|
test "dense BF16 projection source is deterministic and exact" {
  let source = dense_projection_bf16_source(
    input_width=16,
    output_width=128,
    maximum_rows=32,
    function_symbol="lunaflux_advanced_decoder_language_model_head_bf16_v1",
  )
  assert_eq(source.operation(), DenseProjectionBf16)
  assert_eq(source.numerical_contract(), OrderedFloat32AccumulationBf16RoundV1)
  assert_eq(source.input_width(), 16)
  assert_eq(source.output_width(), 128)
  assert_eq(source.maximum_rows(), 32)
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("const int row_count = counts[3];"))
  assert_true(text.contains("row_count > 32"))
  assert_true(text.contains("__fmul_rn"))
  assert_true(text.contains("__fadd_rn"))
  assert_true(text.contains("__float2bfloat16_rn"))
  assert_false(text.contains("bias"))
  let repeated = dense_projection_bf16_source(
    input_width=16,
    output_width=128,
    maximum_rows=32,
    function_symbol="lunaflux_advanced_decoder_language_model_head_bf16_v1",
  )
  assert_eq(source.source_bytes(), repeated.source_bytes())
}

///|
test "dense BF16 projection source validation fails closed" {
  let cases = [
    (0, 1, 1, "valid_symbol"),
    (1, 0, 1, "valid_symbol"),
    (1, 1, 0, "valid_symbol"),
    (1, 1, 1, "invalid-symbol"),
    (1, 1, 1, "1invalid_symbol"),
  ]
  for index, entry in cases {
    let (input_width, output_width, maximum_rows, function_symbol) = entry
    try
      dense_projection_bf16_source(
        input_width~,
        output_width~,
        maximum_rows~,
        function_symbol~,
      )
    catch {
      InvalidInputWidth if index == 0 => ()
      InvalidOutputWidth if index == 1 => ()
      InvalidMaximumRows if index == 2 => ()
      InvalidFunctionSymbol if index == 3 || index == 4 => ()
      _ => fail("unexpected source validation rejection")
    } noraise {
      _ => fail("invalid projection source geometry was admitted")
    }
  }
}

///|
test "fixed-row F32 bias projection source is deterministic and exact" {
  let source = dense_projection_f32_bias_f32_source(
    rows=207L,
    input_width=5376,
    output_width=32,
    function_symbol="lunaflux_joint_diffusion_audio_output_projection_f32_v1",
  )
  assert_eq(source.operation(), DenseProjectionF32BiasF32)
  assert_eq(source.numerical_contract(), OrderedFloat32AccumulationF32OutputV1)
  assert_eq(source.row_count(), 207L)
  assert_eq(source.input_width(), 5376)
  assert_eq(source.output_width(), 32)
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("if (index >= 6624ULL) return;"))
  assert_true(text.contains("const float *bias"))
  assert_true(text.contains("float *output"))
  assert_true(text.contains("__fmul_rn"))
  assert_true(text.contains("__fadd_rn"))
  assert_false(text.contains("cuda_bf16"))
  assert_false(text.contains("counts"))
  let repeated = dense_projection_f32_bias_f32_source(
    rows=207L,
    input_width=5376,
    output_width=32,
    function_symbol="lunaflux_joint_diffusion_audio_output_projection_f32_v1",
  )
  assert_eq(source.source_bytes(), repeated.source_bytes())
}

///|
test "fixed-row F32 bias projection source validation fails closed" {
  let cases = [
    (0L, 1, 1, "valid_symbol"),
    (2147483648L, 1, 1, "valid_symbol"),
    (1L, 0, 1, "valid_symbol"),
    (1L, 1, 0, "valid_symbol"),
    (1L, 1, 1, "invalid-symbol"),
  ]
  for index, entry in cases {
    let (rows, input_width, output_width, function_symbol) = entry
    try
      dense_projection_f32_bias_f32_source(
        rows~,
        input_width~,
        output_width~,
        function_symbol~,
      )
    catch {
      InvalidRowCount if index == 0 || index == 1 => ()
      InvalidInputWidth if index == 2 => ()
      InvalidOutputWidth if index == 3 => ()
      InvalidFunctionSymbol if index == 4 => ()
      _ => fail("unexpected fixed-row source validation rejection")
    } noraise {
      _ => fail("invalid fixed-row projection source geometry was admitted")
    }
  }
}

///|
test "fixed-row F32 bias to BF16 projection source is exact" {
  let source = dense_projection_f32_bias_bf16_source(
    rows=37296L,
    input_width=96,
    output_width=5376,
    function_symbol="lunaflux_joint_diffusion_video_latent_projection_bf16_v1",
  )
  assert_eq(source.operation(), DenseProjectionF32BiasBf16)
  assert_eq(
    source.numerical_contract(),
    OrderedFloat32BiasAccumulationBf16RoundV1,
  )
  assert_eq(source.row_count(), 37296L)
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("if (index >= 200503296ULL) return;"))
  assert_true(text.contains("const float *bias"))
  assert_true(text.contains("__nv_bfloat16 *output"))
  assert_true(text.contains("__float2bfloat16_rn(accumulator)"))
  assert_false(text.contains("counts"))
}
