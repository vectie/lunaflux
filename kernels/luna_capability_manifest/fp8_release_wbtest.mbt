///|
fn fp8_release_manifest_fixture(
  source? : String = "1".repeat(64),
  recipe? : String = "2".repeat(64),
  supports_cublas_lt? : Bool = true,
) -> LunaFp8ReleaseManifestV2 {
  let family = try! @catalog.AotKernelFamily::new(
    module_digest=@catalog.AotArtifactDigest::from_sha256("3".repeat(64)),
    family_id=@catalog.AotKernelFamilyId::from_stable_int(4),
  )
  let entry = try! @catalog.AotKernelEntryPoint::new(
    family~,
    entry_point_id=@catalog.AotKernelEntryPointId::from_stable_int(2),
  )
  let operands : Array[@launch_contract.AotOperand] = [
    try! @launch_contract.AotOperand::new(
      role=RuntimeInput(StepCounts),
      byte_count=20L,
      alignment=4L,
    ),
    try! @launch_contract.AotOperand::new(
      role=Workspace,
      byte_count=4L,
      alignment=4L,
    ),
  ]
  let operation = try! LunaFp8ReleaseOperationV2::new(
    operation=@plan.OperationId::new(2),
    entry_point=entry,
    module_locator="sha256/" + "3".repeat(64) + ".cubin",
    function_symbol="fp8_fixture",
    source_digest_sha256=source,
    recipe_digest_sha256=recipe,
    dimensions=@launch_contract.AotLaunchDimensions::new(
      grid_x=1,
      grid_y=1,
      grid_z=1,
      block_x=256,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    operands~,
    workspace_bytes=4L,
    workspace_alignment=4L,
  )
  operands.push(
    try! @launch_contract.AotOperand::new(
      role=Workspace,
      byte_count=8L,
      alignment=4L,
    ),
  )
  try! LunaFp8ReleaseManifestV2::new(
    identity=try! @spec.ModelIdentity::new(
      @spec.ContentDigest::from_sha256("a".repeat(64)),
      @spec.PlanDigest::from_sha256("b".repeat(64)),
    ),
    target=try! @catalog.DeviceTarget::new(
      compute_major=8,
      compute_minor=9,
      supports_bf16=true,
      supports_cublas_lt~,
    ),
    runtime_recipe_digest_sha256="4".repeat(64),
    operations=[operation],
    maximum_canonical_bytes=16384,
  )
}

///|
test "FP8 release approval is canonical, detached, and externally joined" {
  let first = fp8_release_manifest_fixture()
  let repeated = fp8_release_manifest_fixture()
  assert_eq(first.signing_bytes(), repeated.signing_bytes())
  assert_eq(first.digest(), repeated.digest())
  assert_eq(first.operations()[0].operands().length(), 2)
  assert_true(
    first.digest() !=
    fp8_release_manifest_fixture(source="5".repeat(64)).digest(),
  )
  assert_true(
    first.digest() !=
    fp8_release_manifest_fixture(supports_cublas_lt=false).digest(),
  )
  let key = try! LunaSignatureKeyIdentityDigest::from_sha256("6".repeat(64))
  let signature = try! LunaDetachedSignatureDigest::from_sha256("7".repeat(64))
  let envelope = LunaDetachedSignatureEnvelope::new(
    algorithm=LunaExternalEd25519V1,
    key_identity=key,
    signature_digest=signature,
  )
  let admitted = try! admit_luna_external_approved_fp8_release_v2(
    first,
    envelope,
    luna_test_authenticated_external_approval(
      first.signing_bytes(),
      envelope,
      LunaApprovedImmutableSourceIdentity::from_sha256("8".repeat(64)),
    ),
  )
  assert_eq(admitted.manifest().digest(), first.digest())
  try
    admit_luna_external_approved_fp8_release_v2(
      first,
      envelope,
      luna_test_authenticated_external_approval(
        fp8_release_manifest_fixture(recipe="9".repeat(64)).signing_bytes(),
        envelope,
        LunaApprovedImmutableSourceIdentity::from_sha256("8".repeat(64)),
      ),
    )
  catch {
    Mismatch(Digest) => ()
    _ => fail("unexpected FP8 approval substitution rejection")
  } noraise {
    _ => fail("foreign FP8 release approval was admitted")
  }
}
