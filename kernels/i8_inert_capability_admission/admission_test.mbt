///|
test "catalog-only I8 admission derives exact immutable evidence" {
  let plan = i8_admission_plan(1)
  let resolved = i8_admission_resolved(plan)
  let admitted = i8_admit_fixture(plan, resolved)
  assert_eq(
    admitted.version(),
    @i8_inert_capability_admission.I8InertKernelCapabilityAdmissionVersion::v1(),
  )
  assert_eq(admitted.model_identity(), plan.identity())
  assert_eq(admitted.numeric_schema_digest(), plan.numeric_binding().digest())
  assert_eq(admitted.device_numeric_digest(), i8_admission_device().digest())
  assert_eq(admitted.target(), i8_admission_target())
  assert_eq(admitted.catalog_version(), @catalog.CatalogVersion::v4())
  assert_eq(admitted.operations().length(), 2)
  let gated = admitted.operations()[0]
  assert_eq(gated.operation().as_int(), 1)
  assert_true(gated.operation_kind() is GatedMlp)
  assert_eq(gated.capability(), @plan.KernelCapabilityId::gated_mlp())
  assert_eq(gated.execution_digest(), i8_admission_operation_digest(plan, 1))
  assert_eq(gated.semantic_version(), @catalog.KernelSemanticVersion::v4())
  assert_eq(gated.entry_point().family().family_id().as_int(), 2)
  assert_eq(gated.entry_point().entry_point_id().as_int(), 102)
  assert_eq(gated.workspace_bytes(), 32L)
  assert_eq(gated.workspace_alignment(), 16L)
  assert_eq(gated.weights().length(), 3)
  assert_eq(gated.weights()[0].shape().dimensions(), [32L, 16L])
  assert_eq(gated.weights()[0].scale_shape().dimensions(), [32L])
  assert_eq(gated.weights()[0].output_channel_count(), 32L)
  assert_eq(gated.weights()[2].shape().dimensions(), [16L, 32L])
  assert_eq(gated.weights()[2].output_channel_count(), 16L)
  let projection = admitted.operations()[1]
  assert_true(projection.operation_kind() is OutputProjection)
  assert_eq(projection.weights().length(), 1)
  assert_false(
    gated.weights()[0].scale().tensor() == gated.weights()[1].scale().tensor(),
  )
  assert_false(
    gated.weights()[2].scale().tensor() ==
    projection.weights()[0].scale().tensor(),
  )
  assert_eq(admitted.digest().as_hex().length(), 64)
  assert_eq(admitted.canonical_bytes().length(), 1721)
}

///|
test "catalog-only I8 admission is deterministic and catalog order neutral" {
  let plan = i8_admission_plan(1)
  let first = i8_admit_fixture(plan, i8_admission_resolved(plan))
  let repeated = i8_admit_fixture(plan, i8_admission_resolved(plan))
  let reversed = i8_admit_fixture(
    plan,
    i8_admission_resolved(plan, reverse_catalog=true),
  )
  assert_eq(first.digest(), repeated.digest())
  assert_eq(first.canonical_bytes(), repeated.canonical_bytes())
  assert_eq(first.digest(), reversed.digest())
}

///|
test "every bound model device module and entry identity changes admission" {
  let plan = i8_admission_plan(1)
  let base = i8_admit_fixture(plan, i8_admission_resolved(plan))
  let module_changed = i8_admit_fixture(
    plan,
    i8_admission_resolved(
      plan,
      module_digest="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    ),
  )
  let entry_changed = i8_admit_fixture(
    plan,
    i8_admission_resolved(plan, entry_offset=1000),
  )
  let schema_plan = i8_admission_plan(2)
  let schema_changed = i8_admit_fixture(
    schema_plan,
    i8_admission_resolved(schema_plan),
  )
  let identity_plan = i8_admission_plan(1, identity_seed="foreign")
  let identity_changed = i8_admit_fixture(
    identity_plan,
    i8_admission_resolved(identity_plan),
  )
  let cublas_device = i8_admission_device(cublas_lt=false)
  let target_changed = i8_admit_fixture(
    plan,
    i8_admission_resolved(plan, cublas_lt=false),
    device=cublas_device,
  )
  assert_false(base.digest() == module_changed.digest())
  assert_false(base.digest() == entry_changed.digest())
  assert_false(base.digest() == schema_changed.digest())
  assert_false(base.digest() == identity_changed.digest())
  assert_false(base.digest() == target_changed.digest())
}

///|
test "foreign device model and legacy catalog evidence fail closed" {
  let plan = i8_admission_plan(1)
  let resolved = i8_admission_resolved(plan)
  try
    @i8_inert_capability_admission.admit_catalog_only_symmetric_i8_weight_only_v1(
      plan,
      i8_admission_fp8_device(),
      resolved,
      i8_admission_limits(),
    )
  catch {
    I8InertKernelCapabilityRejected(DeviceCapability) => ()
    _ => fail("unexpected FP8 device substitution error")
  } noraise {
    _ => fail("FP8 device evidence entered I8 admission")
  }
  try
    @i8_inert_capability_admission.admit_catalog_only_symmetric_i8_weight_only_v1(
      plan,
      i8_admission_device(major=9, minor=0),
      resolved,
      i8_admission_limits(),
    )
  catch {
    I8InertKernelCapabilityRejected(DeviceTarget) => ()
    _ => fail("unexpected target substitution error")
  } noraise {
    _ => fail("foreign target evidence entered I8 admission")
  }
  let foreign = i8_admission_plan(1, identity_seed="foreign")
  try
    @i8_inert_capability_admission.admit_catalog_only_symmetric_i8_weight_only_v1(
      foreign,
      i8_admission_device(),
      resolved,
      i8_admission_limits(),
    )
  catch {
    I8InertKernelCapabilityRejected(ModelIdentity) => ()
    _ => fail("unexpected model substitution error")
  } noraise {
    _ => fail("foreign model entered I8 admission")
  }
  let bf16 = i8_admission_plan(3)
  let legacy = i8_admission_legacy_resolved(bf16)
  try
    @i8_inert_capability_admission.admit_catalog_only_symmetric_i8_weight_only_v1(
      plan,
      i8_admission_device(),
      legacy,
      i8_admission_limits(),
    )
  catch {
    I8InertKernelCapabilityRejected(ModelIdentity) => ()
    _ => fail("unexpected legacy-catalog error")
  } noraise {
    _ => fail("foreign legacy-plan catalog entered I8 admission")
  }
}

///|
test "missing I8 selection and bounded envelopes are rejected" {
  let bf16 = i8_admission_plan(3)
  try
    @i8_inert_capability_admission.admit_catalog_only_symmetric_i8_weight_only_v1(
      bf16,
      i8_admission_device(),
      i8_admission_resolved(bf16),
      i8_admission_limits(),
    )
  catch {
    I8InertKernelCapabilityRejected(MissingI8Operation) => ()
    _ => fail("unexpected no-I8 error")
  } noraise {
    _ => fail("empty I8 selection was admitted")
  }
  let plan = i8_admission_plan(1)
  let resolved = i8_admission_resolved(plan)
  let limits = [
    i8_admission_limits(max_operations=1),
    i8_admission_limits(max_operations=2, max_weights=2),
  ]
  for limit in limits {
    try
      @i8_inert_capability_admission.admit_catalog_only_symmetric_i8_weight_only_v1(
        plan,
        i8_admission_device(),
        resolved,
        limit,
      )
    catch {
      I8InertKernelCapabilityRejected(rule) => assert_eq(rule, Limit)
    } noraise {
      _ => fail("undersized I8 admission envelope was ignored")
    }
  }
}
