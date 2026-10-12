///|
test "matrix FMA and butterfly reach both initial softmax and iterative Sinkhorn" {
  let initial = @decoder_hyper_connection_cuda_source.staged_function_and_collapse_bf16_f32_source(
    stream_width=4,
    hidden_width=128,
    maximum_rows=3,
    control_epsilon=0.000001,
    normalization_epsilon=0.000001,
    function_symbol="matrix_control",
    control_matrix_affine_law=FusedMultiplyAdd,
    matrix_sum_law=ButterflyFloat32,
  )
  let source = @utf8.decode(initial.source_bytes())
  assert_true(
    source.contains(
      "__fmaf_rn(mixes[control_index], scale[2], base[control_index])",
    ),
  )
  assert_true(
    source.contains("lf_hc_stage_sum(initial_combination + matrix_base"),
  )
  assert_false(source.contains("__expf"))
  assert_true(
    initial.numerical_contract() is RefinedControlFunctionBf16CollapseV2(..),
  )
  let iterative = @decoder_hyper_connection_cuda_source.staged_positive_sinkhorn_f32_source(
    stream_width=4,
    maximum_rows=3,
    sinkhorn_iterations=20,
    control_epsilon=0.000001,
    function_symbol="matrix_iterative",
    sum_law=ButterflyFloat32,
  )
  assert_eq(
    iterative.numerical_contract(),
    ExplicitF32PositiveSinkhornV1(sum=ButterflyFloat32),
  )
  let text = @utf8.decode(iterative.source_bytes())
  assert_eq(text.split("const float sum = lf_hc_sinkhorn_sum").count(), 4)
  assert_true(text.contains("span = 2"))
  assert_true(text.contains("__fdiv_rn(combination[index], denominator)"))
}
