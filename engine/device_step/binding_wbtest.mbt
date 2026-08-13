///|
test "startup binds only an exact paged model semantic envelope" {
  let paged = step_test_model_plan()
  let limits = step_device_limits(
    max_rows=1,
    max_sequence_tokens=64,
    max_token_id=127,
  )
  let required = validate_plan_binding(paged, step_test_layout(), limits)
  assert_true(
    required.contains(@plan.KernelCapabilityId::positioned_rotary_embedding()),
  )
  assert_true(
    required.contains(@plan.KernelCapabilityId::paged_causal_attention()),
  )
  assert_false(required.contains(@plan.KernelCapabilityId::causal_attention()))
  let stateless = @llama.build(step_test_metadata())
  let stateless_layout = @device_layout.plan(
    stateless.identity(),
    stateless.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=4,
      total_page_count=64,
      max_arena_bytes=1048576L,
    ),
  )
  try validate_plan_binding(stateless, stateless_layout, limits) catch {
    InvalidModelPlan => ()
    _ => fail("unexpected stateless-plan failure")
  } noraise {
    _ => fail("expected stateless-plan rejection")
  }
  try
    validate_plan_binding(
      paged,
      step_test_layout(),
      step_device_limits(max_rows=1, max_sequence_tokens=65, max_token_id=127),
    )
  catch {
    InvalidLimit(SequenceTokens) => ()
    _ => fail("unexpected plan-envelope failure")
  } noraise {
    _ => fail("expected plan-envelope rejection")
  }
}

///|
test "fresh and replacement sequence domains require exact predecessor" {
  validate_next_sequence(false, 0UL, 1UL)
  validate_next_sequence(true, 41UL, 42UL)
  for rejected in [41UL, 43UL] {
    try validate_next_sequence(true, 41UL, rejected) catch {
      InvalidPlan(PlanSequence) => ()
      _ => fail("unexpected replacement-sequence failure")
    } noraise {
      _ => fail("expected replacement-sequence rejection")
    }
  }
  try validate_predecessor_seed(true, 18446744073709551615UL) catch {
    InvalidPlan(PlanSequence) => ()
    _ => fail("unexpected exhausted-predecessor failure")
  } noraise {
    _ => fail("expected exhausted-predecessor rejection")
  }
}
