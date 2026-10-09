///|
test "fixed-row BF16 affine modulation stages every BF16 round before F32 output" {
  let rendered = bf16_shift_scale_f32_fixed_rows_source(
    rows=3L,
    width=4,
    function_symbol="affine_test_v1",
  )
  assert_eq(rendered.operation(), Bf16ShiftScaleToF32)
  assert_eq(rendered.numerical_contract(), StagedBf16ShiftScaleF32OutputV1)
  assert_eq(rendered.row_count(), 3L)
  assert_eq(rendered.width(), 4)
  let source = try! @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("const __nv_bfloat16 *normalized"))
  assert_true(source.contains("const __nv_bfloat16 *shift"))
  assert_true(source.contains("const __nv_bfloat16 *scale"))
  assert_true(source.contains("float *output"))
  assert_true(source.contains("one_plus_scale = __float2bfloat16_rn(__fadd_rn"))
  assert_true(source.contains("scaled = __float2bfloat16_rn(__fmul_rn"))
  assert_true(source.contains("modulated = __float2bfloat16_rn(__fadd_rn"))
  assert_true(source.contains("output[i] = __bfloat162float(modulated)"))
}

///|
test "F32 channel-major destandardization binds exact ordered four-operand ABI" {
  let rendered = f32_channel_major_destandardization_source(
    batch_count=1,
    channels=3,
    elements_per_channel=5L,
    function_symbol="destandardize_v1",
  )
  assert_eq(rendered.operation(), F32ChannelMajorDestandardization)
  assert_eq(rendered.numerical_contract(), OrderedF32ChannelMajorMultiplyAddV1)
  assert_eq(rendered.width(), 3)
  assert_eq(rendered.row_count(), 5L)
  let source = try! @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("const float *normalized"))
  assert_true(source.contains("const float *standard_deviations"))
  assert_true(source.contains("const float *means"))
  assert_true(source.contains("float *output"))
  assert_true(source.contains("(i / 5ULL) % 3ULL"))
  assert_true(source.contains("__fmul_rn"))
  assert_true(source.contains("__fadd_rn"))
}

///|
test "F32 channel-major destandardization rejects hostile geometry and symbol" {
  for hostile in [(0, 1L), (1, 0L), (1, -1L)] {
    let (channels, elements) = hostile
    try
      f32_channel_major_destandardization_source(
        batch_count=1,
        channels~,
        elements_per_channel=elements,
        function_symbol="ok",
      )
    catch {
      InvalidWidth | InvalidRowCount => ()
      _ => fail("unexpected channel affine rejection")
    } noraise {
      _ => fail("hostile channel affine geometry admitted")
    }
  }
  try
    f32_channel_major_destandardization_source(
      batch_count=1,
      channels=2,
      elements_per_channel=4611686018427387904L,
      function_symbol="ok",
    )
  catch {
    DimensionOverflow => ()
    _ => fail("unexpected channel affine overflow rejection")
  } noraise {
    _ => fail("overflowing channel affine geometry admitted")
  }
  try
    f32_channel_major_destandardization_source(
      batch_count=1,
      channels=1,
      elements_per_channel=1L,
      function_symbol="9bad",
    )
  catch {
    InvalidFunctionSymbol => ()
    _ => fail("unexpected channel affine symbol rejection")
  } noraise {
    _ => fail("hostile channel affine symbol admitted")
  }
}

///|
test "F32 channel-major destandardization binds batch and rejects invalid batch" {
  let rendered = try! f32_channel_major_destandardization_source(
    batch_count=2,
    channels=32,
    elements_per_channel=207L,
    function_symbol="audio_destandardize",
  )
  assert_true(rendered.source_bytes().length() > 0)
  try
    f32_channel_major_destandardization_source(
      batch_count=0,
      channels=32,
      elements_per_channel=207L,
      function_symbol="audio_destandardize",
    )
  catch {
    InvalidWidth => ()
    _ => fail("wrong invalid-batch rejection")
  } noraise {
    _ => fail("zero batch admitted")
  }
}

///|
test "affine source rejects hostile geometry and symbols" {
  for rows in [0L, -1L, 2147483648L] {
    try
      bf16_shift_scale_f32_fixed_rows_source(
        rows~,
        width=4,
        function_symbol="ok",
      )
    catch {
      InvalidRowCount => ()
      _ => fail("unexpected row rejection")
    } noraise {
      _ => fail("hostile rows were accepted")
    }
  }
  try
    bf16_shift_scale_f32_fixed_rows_source(
      rows=1L,
      width=0,
      function_symbol="ok",
    )
  catch {
    InvalidWidth => ()
    _ => fail("unexpected width rejection")
  } noraise {
    _ => fail("zero width was accepted")
  }
  try
    bf16_shift_scale_f32_fixed_rows_source(
      rows=1L,
      width=4,
      function_symbol="bad-name",
    )
  catch {
    InvalidFunctionSymbol => ()
    _ => fail("unexpected symbol rejection")
  } noraise {
    _ => fail("hostile symbol was accepted")
  }
  try
    bf16_shift_scale_f32_fixed_rows_source(
      rows=1L,
      width=4,
      function_symbol="9bad",
    )
  catch {
    InvalidFunctionSymbol => ()
    _ => fail("unexpected leading-digit symbol rejection")
  } noraise {
    _ => fail("leading-digit symbol was accepted")
  }
}

///|
test "final AdaLN parameter row binds exact F32 SiLU BF16 dense split ABI" {
  let rendered = final_adaln_parameter_row_source(
    input_width=4,
    hidden_size=2,
    function_symbol="parameter_row_v1",
  )
  assert_eq(rendered.operation(), FinalAdaLnParameterRow)
  assert_eq(
    rendered.numerical_contract(),
    SiluF32CastBf16OrderedDenseBf16OutputV1,
  )
  assert_eq(rendered.row_count(), 1L)
  assert_eq(rendered.width(), 2)
  let source = try! @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("const float *temb"))
  assert_true(source.contains("expf(-x)"))
  assert_true(source.contains("activated_bf16 = __float2bfloat16_rn"))
  assert_true(source.contains("const __nv_bfloat16 *weight"))
  assert_true(source.contains("const __nv_bfloat16 *bias"))
  assert_true(source.contains("__fmul_rn"))
  assert_true(source.contains("__fadd_rn"))
  assert_true(source.contains("shift[output_index]"))
  assert_true(source.contains("scale[output_index - 2]"))
}

///|
test "final AdaLN parameter row rejects output-width overflow before rendering" {
  try
    final_adaln_parameter_row_source(
      input_width=1,
      hidden_size=1073741824,
      function_symbol="overflow_must_not_render",
    )
  catch {
    DimensionOverflow => ()
    _ => fail("unexpected oversized parameter-row rejection")
  } noraise {
    _ => fail("2 * hidden_size overflow was admitted")
  }
}

///|
test "multirow final AdaLN parameters preserve row-major tables and BF16 rounds" {
  let rendered = final_adaln_parameter_rows_source(
    rows=3,
    input_width=4,
    hidden_size=2,
    function_symbol="parameter_rows_v1",
  )
  assert_eq(rendered.row_count(), 3L)
  assert_eq(rendered.width(), 2)
  assert_eq(
    rendered.numerical_contract(),
    SiluF32CastBf16OrderedDenseBf16OutputV1,
  )
  let source = try! @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("const float *temb_rows"))
  assert_true(source.contains("input_base = row * 4ULL"))
  assert_true(source.contains("output_base = row * 2ULL"))
  assert_true(source.contains("activated_bf16 = __float2bfloat16_rn"))
  assert_true(source.contains("shift_rows[output_base + output_index]"))
  assert_true(source.contains("scale_rows[output_base + output_index - 2]"))
}

///|
test "multirow final AdaLN parameters reject hostile row geometry" {
  for rows in [0, -1, 1073741824] {
    try
      final_adaln_parameter_rows_source(
        rows~,
        input_width=1,
        hidden_size=2,
        function_symbol="parameter_rows",
      )
    catch {
      InvalidRowCount | DimensionOverflow => ()
      _ => fail("unexpected multirow rejection")
    } noraise {
      _ => fail("hostile multirow geometry admitted")
    }
  }
}

///|
test "packed modality gather renders exact eight-operand BF16 copy ABI" {
  let rendered = packed_modality_adaln_gather_source(
    packed_rows=9,
    video_rows=3,
    audio_rows=2,
    hidden_size=4,
    function_symbol="packed_gather_v1",
  )
  assert_eq(rendered.operation(), PackedModalityAdaLnGather)
  assert_eq(rendered.numerical_contract(), BitExactBf16PackedModalityGatherV1)
  let source = try! @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("const int32_t *video_indices"))
  assert_true(source.contains("const int32_t *audio_indices"))
  assert_true(source.contains("__nv_bfloat16 *audio_scale"))
  assert_false(source.contains("invalid"))
  try
    packed_modality_adaln_gather_source(
      packed_rows=4,
      video_rows=3,
      audio_rows=2,
      hidden_size=4,
      function_symbol="bad_geometry",
    )
  catch {
    InvalidRowCount => ()
    _ => fail("unexpected gather geometry rejection")
  } noraise {
    _ => fail("oversubscribed modality rows admitted")
  }
}
