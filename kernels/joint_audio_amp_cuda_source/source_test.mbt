///|
test "residual and three-way average renderer pins ordered F32 arithmetic" {
  let source = @joint_audio_amp_cuda_source.f32_residual_and_three_way_average_source(
    element_count=1059840L,
    residual_symbol="amp_residual_v1",
    average_symbol="amp_average_v1",
  )
  assert_eq(source.element_count(), 1059840L)
  let text = @utf8.decode(source.source_bytes()) catch {
    _ => abort("invalid UTF-8")
  }
  assert_true(text.contains("__fadd_rn"))
  assert_true(text.contains("__fdiv_rn"))
}

///|
test "residual and average renderer rejects zero and hostile symbols" {
  try
    @joint_audio_amp_cuda_source.f32_residual_and_three_way_average_source(
      element_count=0L,
      residual_symbol="residual",
      average_symbol="average",
    )
  catch {
    InvalidElementCount => ()
    _ => fail("wrong zero error")
  } noraise {
    _ => fail("zero admitted")
  }
  try
    @joint_audio_amp_cuda_source.f32_residual_and_three_way_average_source(
      element_count=1L,
      residual_symbol="1residual",
      average_symbol="average",
    )
  catch {
    InvalidFunctionSymbol => ()
    _ => fail("wrong symbol error")
  } noraise {
    _ => fail("leading digit admitted")
  }
}

///|
test "scalar oracle preserves residual and left-associated average order" {
  let residual = try! @joint_audio_amp_cuda_source.f32_residual_add_reference(
    [1.0F, -2.0F],
    [3.0F, 5.0F],
  )
  assert_eq(residual, [4.0F, 3.0F])
  let average = try! @joint_audio_amp_cuda_source.f32_three_way_average_reference(
    [1.0F, 2.0F],
    [4.0F, 5.0F],
    [7.0F, 8.0F],
  )
  assert_eq(average, [4.0F, 5.0F])
  try
    @joint_audio_amp_cuda_source.f32_three_way_average_reference([1.0F], [], [
      1.0F,
    ])
  catch {
    InvalidReferenceShape => ()
    _ => fail("wrong shape error")
  } noraise {
    _ => fail("mismatched vectors admitted")
  }
}
