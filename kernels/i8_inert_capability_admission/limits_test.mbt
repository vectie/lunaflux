///|
fn expect_i8_limits_rejected(
  operations : Int,
  weights : Int,
  canonical : Int,
) -> Unit raise {
  try
    @i8_inert_capability_admission.I8InertKernelCapabilityAdmissionLimits::new(
      max_operations=operations,
      max_weights=weights,
      max_canonical_bytes=canonical,
    )
  catch {
    I8InertKernelCapabilityRejected(Limit) => ()
    _ => fail("unexpected I8 limit validation error")
  } noraise {
    _ => fail("incoherent I8 limit envelope was admitted")
  }
}

///|
test "I8 canonical limits are jointly coherent and overflow safe" {
  expect_i8_limits_rejected(1, 1, 1)
  expect_i8_limits_rejected(1, 1, 832)
  expect_i8_limits_rejected(8, 16, 5704)
  expect_i8_limits_rejected(2, 1, 4096)
  expect_i8_limits_rejected(65537, 1, 67108864)
  expect_i8_limits_rejected(1, 299591, 67108864)
  expect_i8_limits_rejected(2147483647, 2147483647, 2147483647)
}

///|
test "exact canonical size boundary admits without slack" {
  let plan = i8_admission_plan(1)
  let exact = try! @i8_inert_capability_admission.I8InertKernelCapabilityAdmissionLimits::new(
    max_operations=2,
    max_weights=4,
    max_canonical_bytes=1721,
  )
  let admitted = i8_admit_fixture(
    plan,
    i8_admission_resolved(plan),
    limits=exact,
  )
  assert_eq(admitted.canonical_bytes().length(), 1721)
  assert_eq(admitted.digest().as_hex().length(), 64)
}

///|
test "resolved aggregate workspace and association are authenticated" {
  let first = i8_admission_plan(1)
  let resolved = i8_admission_resolved(first)
  assert_eq(resolved.max_workspace_bytes(), 32L)
  assert_eq(resolved.max_workspace_alignment(), 16L)
  ignore(i8_admit_fixture(first, resolved))
  let changed_numeric = i8_admission_plan(2)
  assert_false(first.identity() == changed_numeric.identity())
  try
    @i8_inert_capability_admission.admit_catalog_only_symmetric_i8_weight_only_v1(
      changed_numeric,
      i8_admission_device(),
      resolved,
      i8_admission_limits(),
    )
  catch {
    I8InertKernelCapabilityRejected(ModelIdentity) => ()
    _ => fail("unexpected cross-associated resolved catalog error")
  } noraise {
    _ => fail("resolved catalog was associated with a different numeric plan")
  }
}
