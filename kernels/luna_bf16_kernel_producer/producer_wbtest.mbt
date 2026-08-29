///|
test "two-layer release reuses modules and produces the exact full graph" {
  let plan = producer_fixture_plan()
  let compilations = producer_fixture_compilations()
  let first = produce_luna_bf16_kernel_release(
    plan,
    producer_fixture_layout(),
    producer_fixture_target(),
    compilations.rev(),
    producer_fixture_limits(),
  )
  let second = produce_luna_bf16_kernel_release(
    plan,
    producer_fixture_layout(),
    producer_fixture_target(),
    producer_fixture_compilations(),
    producer_fixture_limits(),
  )
  assert_eq(plan.operations().length(), 21)
  assert_eq(first.catalog().bindings().length(), 21)
  assert_eq(first.contracts().contracts().length(), 21)
  assert_eq(first.bundle().operations().length(), 21)
  assert_eq(first.bundle().modules().length(), 9)
  assert_eq(first.bundle().execution_manifest().operation_count(), 21)
  assert_eq(first.bundle().execution_manifest().module_count(), 9)
  assert_eq(first.bundle().execution_manifest().entry_point_count(), 11)
  assert_eq(
    first.bundle().execution_manifest().bytes(),
    second.bundle().execution_manifest().bytes(),
  )
  assert_eq(
    first.bundle().modules().length(),
    second.bundle().modules().length(),
  )
  for index, module_item in first.bundle().modules() {
    assert_eq(module_item.digest(), second.bundle().modules()[index].digest())
    assert_eq(
      module_item.module_bytes(),
      second.bundle().modules()[index].module_bytes(),
    )
  }
  let rms_first = compilations[1].candidate
  let rms_second = compilations[7].candidate
  guard rms_first is Pointwise(first_lowering) else {
    fail("first RMS candidate was not pointwise")
  }
  let first_recipe = @utf8.decode(first_lowering.recipe_bytes())
  assert_true(
    first_recipe.contains(
      "function_symbol=" + first_lowering.function_symbol() + "\n",
    ),
  )
  guard compilations[2].candidate is Projection(projection_lowering) else {
    fail("QKV candidate was not a projection")
  }
  let projection_recipe = @utf8.decode(projection_lowering.recipe_bytes())
  assert_true(
    projection_recipe.contains(
      "function_symbol=" + projection_lowering.function_symbol() + "\n",
    ),
  )
  assert_eq(
    candidate_source_digest(rms_first),
    candidate_source_digest(rms_second),
  )
  assert_true(
    fixture_candidate_recipe_digest(rms_first) !=
    fixture_candidate_recipe_digest(rms_second),
  )
}

///|
test "operation-specific source cannot substitute for reusable family" {
  try
    produce_luna_bf16_kernel_release(
      producer_fixture_plan(),
      producer_fixture_layout(),
      producer_fixture_target(),
      producer_fixture_compilations(legacy_operation=7),
      producer_fixture_limits(),
    )
  catch {
    Mismatch(ReusableFamily) => ()
    _ => fail("unexpected reusable-family rejection")
  } noraise {
    _ => fail("expected operation-specific module rejection")
  }
}

///|
test "missing and substituted compiler evidence fail before bundle publication" {
  let compilations = producer_fixture_compilations()
  try
    produce_luna_bf16_kernel_release(
      producer_fixture_plan(),
      producer_fixture_layout(),
      producer_fixture_target(),
      compilations[:20],
      producer_fixture_limits(),
    )
  catch {
    Invalid(OperationCoverage) => ()
    _ => fail("unexpected coverage rejection")
  } noraise {
    _ => fail("expected missing-operation rejection")
  }
  let original = compilations[0]
  compilations[0] = LunaBf16OfflineCompilation::new(
    candidate=original.candidate,
    first_module=b"substituted",
    second_module=b"substituted",
    receipt=original.receipt,
  )
  try
    produce_luna_bf16_kernel_release(
      producer_fixture_plan(),
      producer_fixture_layout(),
      producer_fixture_target(),
      compilations,
      producer_fixture_limits(),
    )
  catch {
    Invalid(FinalBinding | Bundle) => ()
    _ => fail("unexpected compiler-substitution rejection")
  } noraise {
    _ => fail("expected compiler-substitution rejection")
  }
}
