///|
test "stable biased top-k fixes lower expert id ties" {
  let source = stable_biased_top_k_i32_source(
    routed_expert_count=256,
    experts_per_token=6,
    maximum_rows=32,
    function_symbol="stable_top_k_i32",
  )
  assert_eq(source.operation(), StableBiasedTopKI32)
  assert_eq(source.numerical_contract(), StableLowerExpertIdTopKV1)
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("choice == best_value && expert < best"))
  assert_true(text.contains("__fadd_rn"))
  assert_true(text.contains("selected_indices"))
  assert_eq(
    source.source_bytes(),
    stable_biased_top_k_i32_source(
      routed_expert_count=256,
      experts_per_token=6,
      maximum_rows=32,
      function_symbol="stable_top_k_i32",
    ).source_bytes(),
  )
}

///|
test "selected weights fix ordered normalization and scaling" {
  let source = selected_expert_weight_finalize_f32_source(
    routed_expert_count=384,
    experts_per_token=6,
    maximum_rows=32,
    normalize=true,
    scaling=2.5,
    function_symbol="selected_weights_f32",
  )
  assert_eq(source.operation(), SelectedExpertWeightFinalizeF32)
  assert_eq(source.numerical_contract(), OrderedF32NormalizeScaleV1)
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("__fadd_rn(sum, value)"))
  assert_true(text.contains("__fdiv_rn(value, sum)"))
  assert_true(text.contains("__fmul_rn(normalized, 2.5f)"))
  assert_true(text.contains("selected_weights"))
}

///|
test "routing source rejects hostile geometry scale and symbol" {
  try
    stable_biased_top_k_i32_source(
      routed_expert_count=5,
      experts_per_token=6,
      maximum_rows=1,
      function_symbol="top_k",
    )
  catch {
    InvalidExpertsPerToken => ()
    _ => fail("unexpected top-k geometry rejection")
  } noraise {
    _ => fail("top-k wider than routed experts was admitted")
  }
  try
    selected_expert_weight_finalize_f32_source(
      routed_expert_count=8,
      experts_per_token=2,
      maximum_rows=1,
      normalize=true,
      scaling=0.0,
      function_symbol="weights",
    )
  catch {
    InvalidScaling => ()
    _ => fail("unexpected scaling rejection")
  } noraise {
    _ => fail("zero routing scale was admitted")
  }
  try
    stable_biased_top_k_i32_source(
      routed_expert_count=8,
      experts_per_token=2,
      maximum_rows=1048577,
      function_symbol="9bad",
    )
  catch {
    InvalidMaximumRows => ()
    _ => fail("unexpected maximum-row rejection")
  } noraise {
    _ => fail("oversized routing row envelope was admitted")
  }
  try
    stable_biased_top_k_i32_source(
      routed_expert_count=8,
      experts_per_token=2,
      maximum_rows=1,
      function_symbol="9bad",
    )
  catch {
    InvalidFunctionSymbol => ()
    _ => fail("unexpected routing symbol rejection")
  } noraise {
    _ => fail("invalid CUDA function symbol was admitted")
  }
}
