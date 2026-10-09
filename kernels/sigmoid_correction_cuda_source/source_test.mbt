///|
test "sigmoid correction source preserves separate score outputs" {
  let rendered = sigmoid_and_correction_f32_source(
    expert_count=288,
    maximum_rows=32,
    function_symbol="router_score",
  )
  assert_eq(rendered.operation(), SigmoidAndCorrectionF32)
  assert_eq(rendered.numerical_contract(), Float32SigmoidThenCorrectionAddV1)
  assert_eq(rendered.expert_count(), 288)
  let source = @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("const float *raw_logits"))
  assert_true(source.contains("const float *correction_bias"))
  assert_true(source.contains("float *scores, float *scores_for_choice"))
  assert_true(source.contains("if (scores == scores_for_choice) return"))
  assert_true(source.contains("scores[index] = sigmoid_score"))
  assert_true(
    source.contains("__fadd_rn(sigmoid_score, correction_bias[expert])"),
  )
  assert_false(source.contains("top_k"))
  assert_false(source.contains("group_scores"))
  assert_false(source.contains("normaliz"))
}

///|
test "sigmoid correction source rejects invalid dimensions and symbols" {
  try
    sigmoid_and_correction_f32_source(
      expert_count=0,
      maximum_rows=1,
      function_symbol="score",
    )
  catch {
    InvalidExpertCount => ()
    _ => fail("wrong expert error")
  } noraise {
    _ => fail("zero experts admitted")
  }
  try
    sigmoid_and_correction_f32_source(
      expert_count=1,
      maximum_rows=0,
      function_symbol="score",
    )
  catch {
    InvalidMaximumRows => ()
    _ => fail("wrong row error")
  } noraise {
    _ => fail("zero rows admitted")
  }
  try
    sigmoid_and_correction_f32_source(
      expert_count=1,
      maximum_rows=1,
      function_symbol="1score",
    )
  catch {
    InvalidFunctionSymbol => ()
    _ => fail("wrong symbol error")
  } noraise {
    _ => fail("invalid symbol admitted")
  }
}
