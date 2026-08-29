///|
fn admit_fp8_release_v5(
  root : @approved_fs.ApprovedRoot,
  claims : ManifestV5Claims,
  recipe : @fp8_runtime_recipe.Fp8ReusablePagedRecipeV3,
  artifacts : @artifact.KernelArtifactBundle,
  approval : @worker_wire.WorkerBootstrapLunaApproval,
  verifier : @luna_capability_manifest.LunaStartupExternalApprovalHandoff?,
  parent_witness : @luna_capability_manifest.LunaParentStartupApprovalWitness?,
  limits : @fp8_release_authority.Fp8ReleaseAuthorityLimits,
) -> @fp8_release_authority.Fp8ReusablePagedEnvelopeAuthorityV3 raise ExecutionManifestFileError {
  let launches = recipe.launches()
  if launches.length() != claims.release.operations.length() {
    raise AdmissionRejected(LunaExternalApproval)
  }
  let operations : Array[@luna_capability_manifest.LunaFp8ReleaseOperationV2] = []
  for index, launch in launches {
    let claim = claims.release.operations[index]
    guard claim.operation == launch.operation() &&
      artifacts.entry_point(launch.entry_point()) is Some(entry) else {
      raise AdmissionRejected(LunaExternalApproval)
    }
    operations.push(
      @luna_capability_manifest.LunaFp8ReleaseOperationV2::new(
        operation=launch.operation(),
        entry_point=launch.entry_point(),
        module_locator=claim.module_locator,
        function_symbol=entry.function_symbol(),
        source_digest_sha256=claim.source_digest,
        recipe_digest_sha256=claim.recipe_digest,
        dimensions=launch.dimensions(),
        operands=launch.operands(),
        workspace_bytes=launch.workspace_bytes(),
        workspace_alignment=launch.workspace_alignment(),
      ) catch {
        _ => raise AdmissionRejected(LunaExternalApproval)
      },
    )
  }
  let manifest = @luna_capability_manifest.LunaFp8ReusablePagedReleaseManifestV3::new(
    identity=recipe.startup().model_plan().identity(),
    target=recipe.startup().kernel_artifacts().manifest().target(),
    reusable_recipe_digest_sha256=recipe.digest().as_hex(),
    profile=recipe.profile(),
    kv_layout_version=recipe.kv_layout().stable_layout_version(),
    tokens_per_page=recipe.kv_layout().tokens_per_page(),
    operations~,
    maximum_canonical_bytes=1048576,
  ) catch {
    _ => raise AdmissionRejected(LunaExternalApproval)
  }
  let key = @luna_capability_manifest.LunaSignatureKeyIdentityDigest::from_sha256(
    approval.key_identity(),
  ) catch {
    _ => raise AdmissionRejected(LunaExternalApproval)
  }
  let signature = @luna_capability_manifest.LunaDetachedSignatureDigest::from_sha256(
    approval.signature_digest(),
  ) catch {
    _ => raise AdmissionRejected(LunaExternalApproval)
  }
  let authenticated = match (verifier, parent_witness) {
    (Some(owner), None) =>
      owner.authenticate_manifest_claim(
        manifest.signing_bytes(),
        manifest_digest_sha256=approval.manifest_digest(),
        key_identity_sha256=approval.key_identity(),
        signature_digest_sha256=approval.signature_digest(),
        approved_source_sha256=approval.approved_source(),
      ) catch {
        _ => raise AdmissionRejected(LunaExternalApproval)
      }
    (None, Some(witness)) =>
      witness.authenticate_manifest_claim(
        manifest.signing_bytes(),
        manifest_digest_sha256=approval.manifest_digest(),
        key_identity_sha256=approval.key_identity(),
        signature_digest_sha256=approval.signature_digest(),
        approved_source_sha256=approval.approved_source(),
      ) catch {
        _ => raise AdmissionRejected(LunaExternalApproval)
      }
    _ => raise AdmissionRejected(LunaExternalApproval)
  }
  let admitted = @luna_capability_manifest.admit_luna_external_approved_fp8_reusable_paged_release_v3(
    manifest,
    @luna_capability_manifest.LunaDetachedSignatureEnvelope::new(
      algorithm=LunaExternalEd25519V1,
      key_identity=key,
      signature_digest=signature,
    ),
    authenticated,
  ) catch {
    _ => raise AdmissionRejected(LunaExternalApproval)
  }
  @fp8_release_authority.admit_reusable_paged_v3(
    root,
    @approved_fs.ApprovedRelativeLocator::new(claims.release.manifest_locator) catch {
      _ => raise AdmissionRejected(LunaExternalApproval)
    },
    admitted,
    recipe,
    limits,
  ) catch {
    _ => raise AdmissionRejected(LunaExternalApproval)
  }
}
