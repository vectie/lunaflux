///|
fn tensor_core_test_limits() -> LunaTensorCorePhysicalEvidenceLimits raise {
  LunaTensorCorePhysicalEvidenceLimits::new(
    maximum_result_bytes=65536,
    maximum_manifest_bytes=262144,
  )
}

///|
fn tensor_core_test_admit(
  fixture : TensorCoreEvidenceCandidateFixture,
  bundle : TensorCoreEvidenceBundle,
) -> AdmittedLunaTensorCorePhysicalEvidence raise {
  admit_luna_tile_tensor_core_physical_evidence(
    fixture.candidate,
    fixture.serial,
    fixture.parallel,
    bundle.result,
    bundle.manifest,
    bundle.outer,
    bundle.outer_digest,
    tensor_core_test_limits(),
  )
}

///|
fn tensor_core_test_reseal(
  result : Bytes,
  manifest : Bytes,
) -> TensorCoreEvidenceBundle {
  let outer = @utf8.encode(
    tensor_core_evidence_sha256(manifest).as_hex() +
    "  FILES.sha256\n" +
    tensor_core_evidence_sha256(result).as_hex() +
    "  RESULT.txt\n",
  )
  { result, manifest, outer, outer_digest: tensor_core_evidence_sha256(outer) }
}

///|
test "exact sealed tensor-core campaign admits and binds without manifest authority" {
  let fixture = tensor_core_evidence_candidate_fixture()
  let bundle = tensor_core_evidence_bundle(fixture)
  inspect(@utf8.decode(bundle.result).split("\n").count(), content="68")
  let evidence = tensor_core_test_admit(fixture, bundle) catch {
    error => fail("baseline tensor-core evidence: \{Repr(error)}")
  }
  assert_eq(evidence.candidate_digest(), fixture.candidate.digest().as_hex())
  assert_eq(
    evidence.source_digest(),
    fixture.candidate.source_digest().as_hex(),
  )
  assert_eq(evidence.serial_oracle_digest(), fixture.serial.digest())
  assert_eq(evidence.parallel_fallback_digest(), fixture.parallel.digest())
  assert_eq(evidence.instruction_family(), "MMA")
  assert_eq(evidence.numeric_case_count(), 131072)
  assert_eq(evidence.registers_per_thread(), 64)
  assert_eq(evidence.canonical_bytes(), bundle.result)
  assert_eq(evidence.digest(), tensor_core_evidence_sha256(bundle.result))
  assert_eq(
    evidence.files_manifest_digest(),
    tensor_core_evidence_sha256(bundle.manifest),
  )
  assert_eq(evidence.outer_seal_digest(), bundle.outer_digest)
  assert_false(evidence.manifest_bindable())
  assert_false(evidence.promotion_authorized())
  let qualified = lower_and_bind_externally_qualified_luna_tile_bf16_mma16x16x16(
    fixture.program,
    fixture.serial,
    fixture.parallel,
    fixture.contract,
    fixture.limits,
    evidence,
  )
  assert_true(qualified.physically_qualified())
  assert_eq(qualified.candidate(), fixture.candidate)
  assert_eq(qualified.evidence().outer_seal_digest(), bundle.outer_digest)
  assert_false(qualified.manifest_bindable())
  assert_false(qualified.promotion_authorized())
}

///|
test "tensor-core admission rejects identity target tool SASS numeric sanitizer and authority substitution" {
  let fixture = tensor_core_evidence_candidate_fixture()
  let baseline = tensor_core_evidence_bundle(fixture)
  let substitutions = [
    (
      "tensor_core_candidate_sha256",
      tensor_core_test_digest("foreign candidate").as_hex(),
    ),
    (
      "parallel_plan_sha256",
      tensor_core_test_digest("foreign fallback").as_hex(),
    ),
    ("target", "sm_121"),
    ("sass_instruction_family", "FFMA"),
    ("registers_per_thread", "129"),
    ("cpu_vs_tensor_core_max_abs_error", "0.002"),
    ("memcheck_errors", "1"),
    ("cleanup_balance", "context1"),
    ("manifest_bindable", "true"),
  ]
  for substitution in substitutions {
    let result = tensor_core_test_replace_result(
      baseline.result,
      substitution.0,
      substitution.1,
    )
    let resealed = tensor_core_test_reseal(result, baseline.manifest)
    try tensor_core_test_admit(fixture, resealed) catch {
      LunaTensorCorePhysicalEvidenceInvalid(_)
      | LunaTensorCorePhysicalEvidenceMismatch(_) => ()
      _ => fail("unexpected tensor-core evidence rejection")
    } noraise {
      _ => fail("hostile tensor-core evidence substitution was admitted")
    }
  }
}

///|
test "tensor-core admission rejects stale outer seal manifest replay and foreign candidate" {
  let fixture = tensor_core_evidence_candidate_fixture()
  let bundle = tensor_core_evidence_bundle(fixture)
  let changed = tensor_core_test_replace_result(
    bundle.result,
    "nvcc_sha256",
    tensor_core_test_digest("changed tool").as_hex(),
  )
  try
    admit_luna_tile_tensor_core_physical_evidence(
      fixture.candidate,
      fixture.serial,
      fixture.parallel,
      changed,
      bundle.manifest,
      bundle.outer,
      bundle.outer_digest,
      tensor_core_test_limits(),
    )
  catch {
    LunaTensorCorePhysicalEvidenceMismatch(Seal) => ()
    _ => fail("unexpected stale outer-seal rejection")
  } noraise {
    _ => fail("mutated result passed the out-of-band outer seal")
  }
  let foreign = tensor_core_evidence_candidate_fixture(tile_count=96L)
  try
    admit_luna_tile_tensor_core_physical_evidence(
      foreign.candidate,
      foreign.serial,
      foreign.parallel,
      bundle.result,
      bundle.manifest,
      bundle.outer,
      bundle.outer_digest,
      tensor_core_test_limits(),
    )
  catch {
    LunaTensorCorePhysicalEvidenceMismatch(
      Candidate
      | Oracle
      | ParallelFallback
    ) => ()
    _ => fail("unexpected foreign-candidate evidence rejection")
  } noraise {
    _ => fail("sealed evidence replayed onto a foreign candidate")
  }
  let manifest = @utf8.encode(
    @utf8.decode(bundle.manifest).replace_all(
      old="measurements/memcheck.log",
      new="measurements/memcheck-replay.log",
    ),
  )
  let resealed = tensor_core_test_reseal(bundle.result, manifest)
  try tensor_core_test_admit(fixture, resealed) catch {
    LunaTensorCorePhysicalEvidenceMismatch(Seal) => ()
    _ => fail("unexpected manifest replay rejection")
  } noraise {
    _ => fail("manifest substitution was admitted")
  }
}

///|
test "qualified binder retains exact serial and SIMT fallbacks" {
  let fixture = tensor_core_evidence_candidate_fixture()
  let evidence = tensor_core_test_admit(
    fixture,
    tensor_core_evidence_bundle(fixture),
  )
  let foreign = tensor_core_evidence_candidate_fixture(tile_count=96L)
  try
    lower_and_bind_externally_qualified_luna_tile_bf16_mma16x16x16(
      foreign.program,
      foreign.serial,
      foreign.parallel,
      foreign.contract,
      foreign.limits,
      evidence,
    )
  catch {
    LunaTensorCorePhysicalEvidenceMismatch(Candidate) => ()
    _ => fail("unexpected qualified fallback-binding rejection")
  } noraise {
    _ => fail("qualified evidence lost its serial/SIMT fallback identity")
  }
}
