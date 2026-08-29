///|
fn fp8_manifest_limits(
  entries? : Int = 8,
  canonical_bytes? : Int = 4096,
) -> @numeric_capability_manifest.Fp8KernelManifestLimits {
  try! @numeric_capability_manifest.Fp8KernelManifestLimits::new(
    max_entries=entries,
    max_canonical_bytes=canonical_bytes,
  )
}

///|
fn admit_fixture(
  fixture : Fp8ManifestFixture,
  claims : ArrayView[@numeric_capability_manifest.Fp8KernelManifestClaim],
) -> @numeric_capability_manifest.AdmittedFp8KernelArtifacts raise @numeric_capability_manifest.Fp8KernelManifestError {
  @numeric_capability_manifest.admit_stateless_fp8_e4m3_w8a8(
    fixture.plan,
    fixture.device_numeric,
    fixture.resolved,
    fixture.launches,
    fixture.artifacts,
    claims,
    fp8_manifest_limits(),
  )
}

///|
test "exact FP8 schema, target, launch, and one artifact bundle are admitted" {
  let fixture = fp8_manifest_fixture()
  let admitted = admit_fixture(fixture, [fixture.claim])
  let manifest = admitted.manifest()
  assert_eq(
    manifest.version(),
    @numeric_capability_manifest.Fp8KernelCapabilityManifestVersion::v1(),
  )
  assert_eq(manifest.scope(), StatelessFullContextV1)
  assert_eq(manifest.model_identity(), fixture.plan.identity())
  assert_eq(
    manifest.numeric_schema_digest(),
    fixture.plan.numeric_binding().digest(),
  )
  assert_eq(manifest.device_numeric_digest(), fixture.device_numeric.digest())
  assert_eq(manifest.target(), fixture.target)
  assert_eq(manifest.entries().length(), 1)
  assert_eq(manifest.entries()[0].operation().as_int(), 1)
  assert_eq(manifest.entries()[0].artifact_module_index(), 0)
  assert_eq(manifest.entries()[0].weights().length(), 1)
  assert_eq(manifest.entries()[0].weights()[0].tensor().as_int(), 1)
  assert_eq(manifest.entries()[0].weights()[0].scale().tensor().as_int(), 2)
  assert_eq(admitted.artifacts().module_count(), 1)
  assert_eq(manifest.digest().as_hex().length(), 64)
  let repeated = fp8_manifest_fixture()
  assert_eq(
    admit_fixture(repeated, [repeated.claim]).manifest().digest(),
    manifest.digest(),
  )
}

///|
test "substituted target evidence fails closed" {
  let fixture = fp8_manifest_fixture()
  let substituted : Fp8ManifestFixture = {
    ..fixture,
    device_numeric: fp8_device_numeric(major=9, minor=0),
  }
  try admit_fixture(substituted, [fixture.claim]) catch {
    InvalidFp8KernelManifest(DeviceTarget) => ()
    _ => fail("unexpected target substitution failure")
  } noraise {
    _ => fail("substituted target evidence was admitted")
  }
}

///|
test "vendor BF16 fallback cannot satisfy an FP8 operation" {
  let fixture = fp8_manifest_fixture()
  let vendor = fp8_resolved_catalog(
    fixture.plan,
    fixture.target,
    vendor_output=true,
  )
  try
    @numeric_capability_manifest.admit_stateless_fp8_e4m3_w8a8(
      fixture.plan,
      fixture.device_numeric,
      vendor,
      fixture.launches,
      fixture.artifacts,
      [fixture.claim],
      fp8_manifest_limits(),
    )
  catch {
    InvalidFp8KernelManifest(VendorOrBf16Fallback) => ()
    _ => fail("unexpected vendor fallback failure")
  } noraise {
    _ => fail("vendor BF16 fallback was admitted as FP8")
  }
}

///|
test "missing, reordered, and execution-substituted claims fail closed" {
  let fixture = fp8_manifest_fixture()
  try admit_fixture(fixture, []) catch {
    InvalidFp8KernelManifest(ClaimCount) => ()
    _ => fail("unexpected missing-claim failure")
  } noraise {
    _ => fail("missing FP8 claim was admitted")
  }
  let operation = try! @plan.OperationId::new(0)
  let wrong_operation = @numeric_capability_manifest.Fp8KernelManifestClaim::new(
    profile=fixture.claim.profile(),
    operation~,
    execution=fixture.claim.execution_digest(),
    entry_point=fixture.claim.entry_point(),
  )
  try admit_fixture(fixture, [wrong_operation]) catch {
    InvalidFp8KernelManifest(ClaimOrder) => ()
    _ => fail("unexpected reordered-claim failure")
  } noraise {
    _ => fail("reordered FP8 claim was admitted")
  }
  let wrong_execution = @numeric_capability_manifest.Fp8KernelManifestClaim::new(
    profile=fixture.claim.profile(),
    operation=fixture.claim.operation(),
    execution=fixture.plan.numeric_binding().operations()[0]
      .execution()
      .digest(),
    entry_point=fixture.claim.entry_point(),
  )
  try admit_fixture(fixture, [wrong_execution]) catch {
    InvalidFp8KernelManifest(ClaimExecution) => ()
    _ => fail("unexpected execution substitution failure")
  } noraise {
    _ => fail("substituted execution digest was admitted")
  }
  let wrong_entry = @numeric_capability_manifest.Fp8KernelManifestClaim::new(
    profile=fixture.claim.profile(),
    operation=fixture.claim.operation(),
    execution=fixture.claim.execution_digest(),
    entry_point=fp8_entry(1, 10),
  )
  try admit_fixture(fixture, [wrong_entry]) catch {
    InvalidFp8KernelManifest(ClaimEntryPoint) => ()
    _ => fail("unexpected entry-point substitution failure")
  } noraise {
    _ => fail("BF16 entry point was substituted for an FP8 claim")
  }
}

///|
test "BF16-only model cannot publish an FP8 kernel manifest" {
  let fixture = fp8_manifest_fixture()
  let plan = fp8_fixture_plan(fp8=false)
  let resolved = fp8_resolved_catalog(plan, fixture.target)
  let launches = fp8_launches(plan, resolved)
  let artifacts = fp8_artifacts(launches)
  try
    @numeric_capability_manifest.admit_stateless_fp8_e4m3_w8a8(
      plan,
      fixture.device_numeric,
      resolved,
      launches,
      artifacts,
      [],
      fp8_manifest_limits(),
    )
  catch {
    InvalidFp8KernelManifest(MissingFp8Operation) => ()
    _ => fail("unexpected BF16-only admission failure")
  } noraise {
    _ => fail("BF16-only model published an FP8 manifest")
  }
}

///|
test "FP8 weight-only activation semantics cannot masquerade as W8A8" {
  let fixture = fp8_manifest_fixture()
  let plan = fp8_weight_only_activation_plan()
  let resolved = fp8_resolved_catalog(plan, fixture.target)
  let launches = fp8_launches(plan, resolved)
  let artifacts = fp8_artifacts(launches)
  let claim = @numeric_capability_manifest.Fp8KernelManifestClaim::new(
    profile=fixture.claim.profile(),
    operation=fixture.claim.operation(),
    execution=plan.numeric_binding().operations()[1].execution().digest(),
    entry_point=fixture.claim.entry_point(),
  )
  try
    @numeric_capability_manifest.admit_stateless_fp8_e4m3_w8a8(
      plan,
      fixture.device_numeric,
      resolved,
      launches,
      artifacts,
      [claim],
      fp8_manifest_limits(),
    )
  catch {
    InvalidFp8KernelManifest(ExecutionContract) => ()
    _ => fail("unexpected FP8 weight-only activation failure")
  } noraise {
    _ => fail("FP8 weight-only activation was mislabeled as W8A8")
  }
}

///|
test "canonical and entry-count limits are enforced before publication" {
  let fixture = fp8_manifest_fixture()
  try
    @numeric_capability_manifest.admit_stateless_fp8_e4m3_w8a8(
      fixture.plan,
      fixture.device_numeric,
      fixture.resolved,
      fixture.launches,
      fixture.artifacts,
      [fixture.claim],
      fp8_manifest_limits(canonical_bytes=1),
    )
  catch {
    InvalidFp8KernelManifest(CanonicalBytes) => ()
    _ => fail("unexpected canonical limit failure")
  } noraise {
    _ => fail("oversized canonical manifest was published")
  }
}
