///|
test "grouped routing source keeps choice and weight score streams distinct" {
  let rendered = grouped_choice_and_selected_weight_f32_source(
    routed_expert_count=288,
    group_count=1,
    group_score_topk=2,
    selected_group_count=1,
    experts_per_token=8,
    normalize=true,
    scaling=2.5,
    maximum_rows=32,
    function_symbol="grouped_route",
  )
  assert_eq(rendered.operation(), GroupedChoiceAndSelectedWeightF32)
  assert_eq(rendered.group_count(), 1)
  let source = @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("const float *uncorrected_scores"))
  assert_true(source.contains("const float *choice_scores"))
  assert_true(source.contains("float group_scores[1]"))
  assert_true(source.contains("int32_t group_top[2]"))
  assert_true(source.contains("int32_t selected_groups[1]"))
  assert_true(source.contains("__fadd_rn(denominator, 1.0e-20f)"))
  assert_true(source.contains("__fmul_rn(normalized, 2.5f)"))
  assert_false(source.contains("expert_output"))
}

///|
test "grouped routing source rejects hostile geometry" {
  try
    grouped_choice_and_selected_weight_f32_source(
      routed_expert_count=288,
      group_count=5,
      group_score_topk=2,
      selected_group_count=1,
      experts_per_token=8,
      normalize=true,
      scaling=2.5,
      maximum_rows=32,
      function_symbol="route",
    )
  catch {
    InvalidGroupGeometry => ()
    _ => fail("wrong group geometry error")
  } noraise {
    _ => fail("non-divisible grouping admitted")
  }
  try
    grouped_choice_and_selected_weight_f32_source(
      routed_expert_count=8,
      group_count=4,
      group_score_topk=2,
      selected_group_count=1,
      experts_per_token=3,
      normalize=true,
      scaling=2.5,
      maximum_rows=32,
      function_symbol="route",
    )
  catch {
    InvalidGroupGeometry => ()
    _ => fail("wrong selected-capacity error")
  } noraise {
    _ => fail("insufficient selected-group capacity admitted")
  }
}
