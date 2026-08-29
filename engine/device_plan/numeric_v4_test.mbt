///|
test "v4 static plan retains exact numeric and catalog evidence" {
  let plan = d2_v4_plan(3)
  let layout = try! @device_materialize.plan_numeric_layout(
    plan,
    alignment_bytes=16L,
    max_arena_bytes=1024L,
  )
  let catalog = d2_v4_catalog(plan)
  let result = try! @device_plan.build(plan, layout, catalog)
  assert_eq(result.identity(), plan.identity())
  assert_eq(result.catalog_version(), @catalog.CatalogVersion::v4())
  assert_eq(result.numeric_schema_digest(), plan.numeric_binding().digest())
  assert_eq(result.weight_layout(), layout)
  assert_eq(
    result.weight_layout().numeric_schema_digest(),
    Some(plan.numeric_binding().digest()),
  )
  for index, operation in result.operations() {
    assert_eq(operation.binding(), catalog.bindings()[index])
    assert_eq(
      operation.operation_execution_digest(),
      d2_v4_operation_digest(plan, index),
    )
    assert_eq(
      operation.binding().operation_execution_digest(),
      Some(operation.operation_execution_digest()),
    )
    assert_eq(
      operation.binding().semantic_version(),
      @catalog.KernelSemanticVersion::v4(),
    )
    assert_true(operation.binding().aot_entry_point() is Some(_))
    assert_eq(
      operation.binding().aot_entry_point(),
      catalog.bindings()[index].aot_entry_point(),
    )
  }
}

///|
test "v4 rejects same-size layout from a different numeric plan" {
  let fp8 = d2_v4_plan(2)
  let i8 = d2_v4_plan(3)
  let foreign_layout = try! @device_materialize.plan_numeric_layout(
    fp8,
    alignment_bytes=16L,
    max_arena_bytes=1024L,
  )
  let exact_layout = try! @device_materialize.plan_numeric_layout(
    i8,
    alignment_bytes=16L,
    max_arena_bytes=1024L,
  )
  assert_eq(foreign_layout.arena_bytes(), exact_layout.arena_bytes())
  assert_eq(
    foreign_layout.materialized_bytes(),
    exact_layout.materialized_bytes(),
  )
  try @device_plan.build(i8, foreign_layout, d2_v4_catalog(i8)) catch {
    IdentityMismatch(WeightPlan) => ()
    _ => fail("unexpected foreign numeric-layout error")
  } noraise {
    _ => fail("same-size foreign numeric layout was accepted")
  }
}

///|
test "v4 rejects catalog resolved for a different numeric plan" {
  let fp8 = d2_v4_plan(2)
  let i8 = d2_v4_plan(3)
  let layout = try! @device_materialize.plan_numeric_layout(
    i8,
    alignment_bytes=16L,
    max_arena_bytes=1024L,
  )
  try @device_plan.build(i8, layout, d2_v4_catalog(fp8)) catch {
    IdentityMismatch(KernelCatalog) => ()
    _ => fail("unexpected foreign numeric-catalog error")
  } noraise {
    _ => fail("foreign operation execution digest was accepted")
  }
}

///|
test "v4 rejects foreign weight content before plan identity" {
  let original = d2_v4_plan(3)
  let changed = d2_v4_plan(
    3,
    content_digest="1111111111111111111111111111111111111111111111111111111111111111",
  )
  assert_eq(original.identity().plan(), changed.identity().plan())
  assert_false(original.identity() == changed.identity())
  let layout = try! @device_materialize.plan_numeric_layout(
    original,
    alignment_bytes=16L,
    max_arena_bytes=1024L,
  )
  try @device_plan.build(changed, layout, d2_v4_catalog(changed)) catch {
    IdentityMismatch(WeightContent) => ()
    _ => fail("unexpected v4 full-identity error")
  } noraise {
    _ => fail("v4 layout plan identity substitution was accepted")
  }
}

///|
test "v4 requires exact full catalog ModelIdentity" {
  let original = d2_v4_plan(3)
  let changed = d2_v4_plan(
    3,
    content_digest="2222222222222222222222222222222222222222222222222222222222222222",
  )
  assert_eq(original.identity().plan(), changed.identity().plan())
  assert_false(original.identity() == changed.identity())
  let layout = try! @device_materialize.plan_numeric_layout(
    changed,
    alignment_bytes=16L,
    max_arena_bytes=1024L,
  )
  try @device_plan.build(changed, layout, d2_v4_catalog(original)) catch {
    IdentityMismatch(KernelCatalog) => ()
    _ => fail("unexpected v4 catalog-identity error")
  } noraise {
    _ => fail("v4 catalog plan identity substitution was accepted")
  }
}

///|
test "v4 rejects legacy weight evidence before numeric-schema binding" {
  let legacy_layout = device_plan_test_layout()
  let plan = d2_v4_plan_with_content(
    3,
    device_plan_test_model().content_digest(),
  )
  // Slice A permits legacy BF16 weight reuse by content only. It does not let
  // that weight-only evidence impersonate the self-authenticated plan.
  assert_eq(legacy_layout.identity().content(), plan.identity().content())
  assert_false(legacy_layout.identity() == plan.identity())
  assert_eq(legacy_layout.numeric_schema_digest(), None)
  try @device_plan.build(plan, legacy_layout, d2_v4_catalog(plan)) catch {
    IdentityMismatch(WeightPlan) => ()
    _ => fail("unexpected legacy-layout v4 error")
  } noraise {
    _ => fail("v4 accepted a layout without a numeric-schema binding")
  }
}

///|
test "legacy execution profile rejects v4 before profile publication" {
  let plan = d2_v4_plan(3)
  let layout = try! @device_materialize.plan_numeric_layout(
    plan,
    alignment_bytes=16L,
    max_arena_bytes=1024L,
  )
  let static_plan = try! @device_plan.build(plan, layout, d2_v4_catalog(plan))
  let memory = try! @device_memory.plan(
    static_plan,
    @device_memory.DeviceMemoryLimits::new(
      activation_alignment=16L,
      max_arena_bytes=4096L,
    ),
  )
  try
    @device_profile.build(
      static_plan,
      memory,
      @device_profile.ExactExecutionShape::full_prefill(
        batch_rows=1,
        sequence_tokens=1,
      ),
    )
  catch {
    InvalidPlan(CatalogVersion) => ()
    _ => fail("unexpected legacy-profile v4 rejection")
  } noraise {
    _ => fail("legacy profile published incomplete v4 numeric evidence")
  }
}
