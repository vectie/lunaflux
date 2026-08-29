///|
test "physical fused evidence admits exact immutable qualification facts without authority" {
  let fixture = fused_physical_black_box_fixture()
  let canonical = fused_physical_test_canonical(fixture)
  let evidence = fused_physical_test_admit(fixture, canonical)
  assert_eq(evidence.canonical_bytes(), canonical)
  assert_eq(evidence.digest(), fused_physical_test_digest(canonical))
  assert_eq(
    evidence.referee_shape_set_digest(),
    fused_physical_test_referee_digest(),
  )
  assert_eq(evidence.qkv_candidate_digest(), fixture.qkv.candidate_digest())
  assert_eq(evidence.qkv_source_digest(), fixture.qkv.source_digest())
  assert_eq(evidence.qkv_recipe_digest(), fixture.qkv.recipe_digest())
  assert_eq(evidence.qkv_cubin_digest(), fixture.qkv_compiled.module_digest())
  assert_eq(
    evidence.qkv_compile_receipt_digest(),
    fixture.qkv_compiled.compile_receipt_digest(),
  )
  assert_eq(evidence.qkv_observed_absolute_error_ppb(), 1000L)
  assert_eq(evidence.qkv_observed_relative_error_ppb(), 2000L)
  assert_eq(evidence.qkv_dispatch_canary(), fixture.qkv.dispatch_canary() * 12U)
  assert_eq(
    evidence.residual_candidate_digest(),
    fixture.residual.candidate_digest(),
  )
  assert_eq(evidence.residual_source_digest(), fixture.residual.source_digest())
  assert_eq(evidence.residual_recipe_digest(), fixture.residual.recipe_digest())
  assert_eq(
    evidence.residual_cubin_digest(),
    fixture.residual_compiled.module_digest(),
  )
  assert_eq(
    evidence.residual_compile_receipt_digest(),
    fixture.residual_compiled.compile_receipt_digest(),
  )
  assert_eq(evidence.residual_observed_absolute_error_ppb(), 500L)
  assert_eq(evidence.residual_observed_relative_error_ppb(), 1000L)
  assert_eq(
    evidence.residual_dispatch_canary(),
    fixture.residual.dispatch_canary() * 12U,
  )
  assert_eq(evidence.artifact_directory_seal_digest().as_hex(), "e".repeat(64))
  assert_eq(evidence.evidence_directory_seal_digest().as_hex(), "f".repeat(64))
  assert_false(evidence.manifest_bindable())
  assert_false(evidence.promotion_authorized())
}

///|
test "physical fused evidence rejects cross-family compiled artifact replay" {
  let fixture = fused_physical_black_box_fixture()
  let canonical = fused_physical_test_canonical(fixture)
  try
    admit_fused_physical_evidence(
      fixture.qkv,
      fixture.residual_compiled,
      fixture.residual,
      fixture.qkv_compiled,
      canonical,
      expected_digest=fused_physical_test_digest(canonical),
      expected_approved_policy_digest=fused_physical_test_policy_digest(),
    )
  catch {
    @luna_fused_physical_evidence.FusedPhysicalEvidenceMismatch(Candidate) => ()
    _ => fail("unexpected cross-family artifact rejection")
  } noraise {
    _ => fail("cross-family compiled artifacts were admitted")
  }
}
