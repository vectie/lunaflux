///|
priv struct ApprovedLunaEvidenceLocators {
  specialization : @approved_fs.ApprovedRelativeLocator
  capability : @approved_fs.ApprovedRelativeLocator
}

///|
fn preflight_luna_approval_binding(
  claim : ManifestLunaAuthorizationClaim?,
  approval : @worker_wire.WorkerBootstrapLunaApproval?,
) -> Unit raise ExecutionManifestFileError {
  match (claim, approval) {
    (None, None) => ()
    (Some(claim), Some(approval)) =>
      if approval.manifest_digest() != claim.capability_digest ||
        approval.key_identity() != claim.key_identity ||
        approval.signature_digest() != claim.signature_digest {
        raise AdmissionRejected(LunaExternalApproval)
      }
    _ => raise AdmissionRejected(LunaExternalApproval)
  }
}

///|
fn validate_luna_evidence_locators(
  claim : ManifestLunaAuthorizationClaim?,
  modules : ArrayView[ManifestModuleClaim],
) -> ApprovedLunaEvidenceLocators? raise ExecutionManifestFileError {
  guard claim is Some(claim) else { return None }
  if claim.specialization_locator == claim.capability_locator {
    raise AdmissionRejected(LunaSpecialization)
  }
  let specialization = @approved_fs.ApprovedRelativeLocator::new(
    claim.specialization_locator,
  ) catch {
    _ => raise AdmissionRejected(LunaSpecialization)
  }
  let capability = @approved_fs.ApprovedRelativeLocator::new(
    claim.capability_locator,
  ) catch {
    _ => raise AdmissionRejected(LunaCapability)
  }
  for module_ in modules {
    if module_.path == claim.specialization_locator ||
      module_.path == claim.capability_locator {
      raise AdmissionRejected(LunaArtifactAuthorization)
    }
  }
  Some({ specialization, capability })
}

///|
fn read_luna_evidence_snapshot(
  root : @approved_fs.ApprovedRoot,
  locator : @approved_fs.ApprovedRelativeLocator,
  expected_digest : String,
  maximum : Int64,
) -> Bytes raise ExecutionManifestFileError {
  let file = root.open_file(locator) catch {
    error => raise map_fs_error(error, Open, maximum)
  }
  let bytes = file.read_immutable_snapshot(maximum_bytes=maximum) catch {
    error => {
      let primary = map_fs_error(error, Snapshot, maximum)
      raise_primary_after_close(primary, file, maximum)
    }
  }
  close_manifest_file(file, maximum)
  if @crypto.bytes_to_hex_string(@crypto.sha256(bytes)) != expected_digest {
    raise DigestMismatch
  }
  bytes
}

///|
fn luna_canonical_limit(
  limits : ExecutionManifestFileLimits,
) -> Int raise ExecutionManifestFileError {
  let value = limits.max_manifest_bytes().to_int()
  if value <= 0 || Int64::from_int(value) != limits.max_manifest_bytes() {
    raise InvalidLimit(ManifestBytes)
  }
  value
}

///|
fn admit_luna_authorization(
  approved_root : @approved_fs.ApprovedRoot,
  claim : ManifestLunaAuthorizationClaim?,
  locators : ApprovedLunaEvidenceLocators?,
  approval : @worker_wire.WorkerBootstrapLunaApproval?,
  verifier : @luna_capability_manifest.LunaStartupExternalApprovalHandoff?,
  parent_witness : @luna_capability_manifest.LunaParentStartupApprovalWitness?,
  model_plan : @plan.ModelPlan,
  static_plan : @device_plan.StaticDevicePlan,
  memory_plan : @device_memory.DeviceMemoryPlan,
  catalog : @catalog.ResolvedKernelCatalog,
  artifacts : @artifact.KernelArtifactBundle,
  contracts : @launch_contract.PagedKvLaunchContractSet,
  limits : ExecutionManifestFileLimits,
) -> @luna_artifact_admission.LunaAotKernelAdmission? raise ExecutionManifestFileError {
  match (claim, locators, approval, verifier, parent_witness) {
    (None, None, None, None, None) => None
    (Some(claim), Some(locators), Some(approval), verifier, parent_witness) => {
      if (verifier is Some(_)) == (parent_witness is Some(_)) {
        raise AdmissionRejected(LunaExternalApproval)
      }
      preflight_luna_approval_binding(Some(claim), Some(approval))
      if claim.specialization_locator == claim.capability_locator {
        raise AdmissionRejected(LunaSpecialization)
      }
      let maximum = limits.max_manifest_bytes()
      let specialization_bytes = read_luna_evidence_snapshot(
        approved_root,
        locators.specialization,
        claim.specialization_digest,
        maximum,
      )
      let capability_bytes = read_luna_evidence_snapshot(
        approved_root,
        locators.capability,
        claim.capability_digest,
        maximum,
      )
      let specialization_digest = @luna_specialization.LunaSpecializationDigest::from_sha256(
        claim.specialization_digest,
      ) catch {
        _ => raise AdmissionRejected(LunaSpecialization)
      }
      let specialization_limits = @luna_specialization.LunaSpecializationLimits::new(
        max_canonical_bytes=luna_canonical_limit(limits),
        max_bound_regions=4,
      ) catch {
        _ => raise AdmissionRejected(LunaSpecialization)
      }
      let decoded = @luna_specialization.admit_persisted_luna_specialization(
        specialization_bytes, specialization_digest, specialization_limits,
      ) catch {
        _ => raise AdmissionRejected(LunaSpecialization)
      }
      let profile = @device_profile.build(
        static_plan,
        memory_plan,
        decoded.profile_shape(),
      ) catch {
        _ => raise AdmissionRejected(LunaExecutionProfile)
      }
      let specialization = @luna_specialization.readmit_persisted_luna_specialization(
        decoded, model_plan, static_plan, profile, catalog,
      ) catch {
        _ => raise AdmissionRejected(LunaSpecialization)
      }
      if specialization.operation_id().as_int() != claim.operation_id {
        raise AdmissionRejected(LunaSpecialization)
      }
      let capability_digest = @luna_capability_manifest.LunaCapabilityManifestDigest::from_sha256(
        claim.capability_digest,
      ) catch {
        _ => raise AdmissionRejected(LunaCapability)
      }
      let entry = @luna_capability_manifest.admit_persisted_luna_capability_manifest(
        capability_bytes,
        capability_digest,
        specialization,
        luna_canonical_limit(limits),
      ) catch {
        _ => raise AdmissionRejected(LunaCapability)
      }
      let key = @luna_capability_manifest.LunaSignatureKeyIdentityDigest::from_sha256(
        claim.key_identity,
      ) catch {
        _ => raise AdmissionRejected(LunaExternalApproval)
      }
      let signature = @luna_capability_manifest.LunaDetachedSignatureDigest::from_sha256(
        claim.signature_digest,
      ) catch {
        _ => raise AdmissionRejected(LunaExternalApproval)
      }
      let authenticated = match (verifier, parent_witness) {
        (Some(owner), None) =>
          owner.authenticate_manifest_claim(
            capability_bytes,
            manifest_digest_sha256=approval.manifest_digest(),
            key_identity_sha256=approval.key_identity(),
            signature_digest_sha256=approval.signature_digest(),
            approved_source_sha256=approval.approved_source(),
          ) catch {
            _ => raise AdmissionRejected(LunaExternalApproval)
          }
        (None, Some(witness)) =>
          witness.authenticate_manifest_claim(
            capability_bytes,
            manifest_digest_sha256=approval.manifest_digest(),
            key_identity_sha256=approval.key_identity(),
            signature_digest_sha256=approval.signature_digest(),
            approved_source_sha256=approval.approved_source(),
          ) catch {
            _ => raise AdmissionRejected(LunaExternalApproval)
          }
        _ => raise AdmissionRejected(LunaExternalApproval)
      }
      let admitted_manifest = @luna_capability_manifest.admit_luna_external_approved_capability(
        entry,
        @luna_capability_manifest.LunaDetachedSignatureEnvelope::new(
          algorithm=LunaExternalEd25519V1,
          key_identity=key,
          signature_digest=signature,
        ),
        authenticated,
      ) catch {
        _ => raise AdmissionRejected(LunaExternalApproval)
      }
      Some(
        @luna_artifact_admission.admit_luna_specialized_artifact(
          admitted_manifest, specialization, artifacts, contracts,
        ) catch {
          _ => raise AdmissionRejected(LunaArtifactAuthorization)
        },
      )
    }
    _ => raise AdmissionRejected(LunaExternalApproval)
  }
}
