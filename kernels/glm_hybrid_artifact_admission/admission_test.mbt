///|
test "outer decoder residual operand ordering fails closed" {
  let requirements = artifact_test_requirements()
  let claims = artifact_test_claims(
    requirements,
    artifact_test_target(),
    qualified=true,
    wrong_residual_operands=true,
  )
  try
    @glm_hybrid_artifact_admission.admit(
      requirements,
      artifact_test_target(),
      claims,
      artifact_test_limits(),
    )
  catch {
    InvalidLaunchOperands(ordinal~) =>
      assert_true(
        requirements.requirements()[ordinal].operation()
        is MoeDecoderResidualAdd(_),
      )
    _ => fail("unexpected residual ABI error")
  } noraise {
    _ => fail("swapped residual and branch roles admitted")
  }
}

///|
test "all caller-declared qualification digests remain non-runnable" {
  let requirements = artifact_test_requirements()
  let target = artifact_test_target()
  let claims = artifact_test_claims(requirements, target, qualified=true)
  let first = @glm_hybrid_artifact_admission.admit(
    requirements,
    target,
    claims,
    artifact_test_limits(),
  )
  let second = @glm_hybrid_artifact_admission.admit(
    requirements,
    target,
    claims,
    artifact_test_limits(),
  )
  assert_eq(first, second)
  assert_eq(first.status(), NonRunnableWithDeclaredQualificationDigests)
  assert_eq(first.artifacts().length(), requirements.requirements().length())
  assert_eq(first.digest().as_hex().length(), 64)
  try first.require_execution_authority() catch {
    NoExecutionAuthority => ()
    _ => fail("unexpected execution-authority error")
  } noraise {
    _ => fail("artifact admission granted execution authority")
  }
}

///|
test "unqualified complete evidence remains explicitly non-runnable" {
  let requirements = artifact_test_requirements()
  let target = artifact_test_target()
  let admitted = @glm_hybrid_artifact_admission.admit(
    requirements,
    target,
    artifact_test_claims(requirements, target, qualified=false),
    artifact_test_limits(),
  )
  assert_eq(admitted.status(), NonRunnableUnqualified)
}

///|
test "missing duplicate target capability and geometry mutations fail closed" {
  let requirements = artifact_test_requirements()
  let target = artifact_test_target()
  let exact = artifact_test_claims(requirements, target, qualified=true)
  try
    @glm_hybrid_artifact_admission.admit(
      requirements,
      target,
      exact[1:].to_owned(),
      artifact_test_limits(),
    )
  catch {
    ClaimCountMismatch(expected=29, actual=28) => ()
    _ => fail("unexpected missing-claim error")
  } noraise {
    _ => fail("missing claim admitted")
  }
  let duplicate = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    duplicate_ordinal=true,
  )
  try
    @glm_hybrid_artifact_admission.admit(
      requirements,
      target,
      duplicate,
      artifact_test_limits(),
    )
  catch {
    DuplicateClaim(ordinal=0) => ()
    _ => fail("unexpected duplicate error")
  } noraise {
    _ => fail("duplicate claim admitted")
  }
  let wrong_target = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    wrong_target=true,
  )
  try
    @glm_hybrid_artifact_admission.admit(
      requirements,
      target,
      wrong_target,
      artifact_test_limits(),
    )
  catch {
    TargetMismatch(ordinal=0) => ()
    _ => fail("unexpected target error")
  } noraise {
    _ => fail("wrong target admitted")
  }
  let wrong_capability = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    wrong_capability=true,
  )
  try
    @glm_hybrid_artifact_admission.admit(
      requirements,
      target,
      wrong_capability,
      artifact_test_limits(),
    )
  catch {
    CapabilityMismatch(ordinal=0) => ()
    _ => fail("unexpected capability error")
  } noraise {
    _ => fail("wrong capability admitted")
  }
  let wrong_geometry = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    wrong_geometry=true,
  )
  try
    @glm_hybrid_artifact_admission.admit(
      requirements,
      target,
      wrong_geometry,
      artifact_test_limits(),
    )
  catch {
    GeometryMismatch(ordinal=1) => ()
    _ => fail("unexpected geometry error")
  } noraise {
    _ => fail("wrong geometry admitted")
  }
}

///|
test "module bytes are not checksummed but source and entrypoint contracts remain checked" {
  let requirements = artifact_test_requirements()
  let target = artifact_test_target()
  let wrong_source = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    wrong_source=true,
  )
  try
    @glm_hybrid_artifact_admission.admit(
      requirements,
      target,
      wrong_source,
      artifact_test_limits(),
    )
  catch {
    SourceDigestMismatch(ordinal=0) => ()
    _ => fail("unexpected source error")
  } noraise {
    _ => fail("wrong source admitted")
  }
  let wrong_bytes = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    wrong_bytes=true,
  )
  let labelled = @glm_hybrid_artifact_admission.admit(
    requirements,
    target,
    wrong_bytes,
    artifact_test_limits(),
  )
  assert_eq(labelled.artifacts().length(), requirements.requirements().length())
  let wrong_identity = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    wrong_identity=true,
  )
  try
    @glm_hybrid_artifact_admission.admit(
      requirements,
      target,
      wrong_identity,
      artifact_test_limits(),
    )
  catch {
    SourceIdentityMismatch(ordinal=0) => ()
    _ => fail("unexpected source-identity error")
  } noraise {
    _ => fail("wrong source identity admitted")
  }
  let wrong_entry_module = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    wrong_entry_module=true,
  )
  try
    @glm_hybrid_artifact_admission.admit(
      requirements,
      target,
      wrong_entry_module,
      artifact_test_limits(),
    )
  catch {
    EntryPointModuleMismatch(ordinal=0) => ()
    _ => fail("unexpected entrypoint-module error")
  } noraise {
    _ => fail("wrong entrypoint module admitted")
  }
  let duplicate_entry = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    duplicate_entry=true,
  )
  try
    @glm_hybrid_artifact_admission.admit(
      requirements,
      target,
      duplicate_entry,
      artifact_test_limits(),
    )
  catch {
    DuplicateEntryPoint(ordinal=1) => ()
    _ => fail("unexpected entrypoint error")
  } noraise {
    _ => fail("duplicate entrypoint admitted")
  }
  try
    @glm_hybrid_artifact_admission.GlmHybridDeclaredQualificationDigest::from_sha256(
      "ABC",
    )
  catch {
    InvalidDeclaredQualificationDigest => ()
    _ => fail("unexpected qualification error")
  } noraise {
    _ => fail("invalid qualification admitted")
  }
}
