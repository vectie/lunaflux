///|
test "step rotary production source binds an explicit prepare-before-borrow ABI" {
  let qualification = qwen_fused_test_candidate(
    head_dimension=128,
    maximum_tokens=32,
  )
  let legacy = lower_fused_qwen_qkv_qknorm_rope_paged_kv_write_production_candidate(
    qualification,
    fused_test_limits(),
  )
  let cached = lower_fused_qwen_qkv_qknorm_rope_paged_kv_write_production_candidate(
    qualification,
    fused_test_limits(),
    step_rotary_cache=true,
  )
  assert_false(legacy.uses_step_rotary_cache())
  assert_true(cached.uses_step_rotary_cache())
  assert_false(legacy.candidate_digest() == cached.candidate_digest())
  let source = @utf8.decode(cached.source_bytes())
  let recipe = @utf8.decode(cached.recipe_bytes())
  assert_true(source.contains("const float2* __restrict__ rotary_pairs"))
  assert_true(
    source.contains("lf_ingress_load_rotary(rotary_pairs, token, lane)"),
  )
  assert_true(
    source.contains("_rotary_prepare(const int32_t* __restrict__ step_counts"),
  )
  assert_eq(source.split("sincosf(").length(), 2)
  assert_false(source.contains("lf_ingress_prepare_rotary(position, lane)"))
  assert_true(recipe.contains("rotary_cache_bytes=16384\n"))
  assert_true(recipe.contains("rotary_prepare_grid=16,1,1\n"))
  assert_true(
    recipe.contains(
      "rotary_prepare_effect=produce-before-all-layer-consumers\n",
    ),
  )
  assert_true(recipe.contains(",rotary_pairs\n"))
}
