///|
/// Independent expected deployment binding for one rooted single-worker
/// service. It remains public for constructor compatibility, but is retained
/// only inside the private physical binding after service preparation.
pub struct WorkerServiceBinding {
  priv bootstrap_digest : @worker_wire.WorkerBootstrapDigest
  priv bootstrap_source_digest : @worker_wire.WorkerBootstrapSourceDigest
  priv device_ordinal : Int
  priv inference_limits : @inference.InferenceLimits
  priv parent_approval : @worker_wire.WorkerParentApprovalSubjects?
} derive(Debug, Eq)

///|
/// Backend-neutral identity retained unchanged across physical worker or group
/// replacement. Device ordinals, rank bootstrap digests, and topology remain
/// inside the selected private physical binding.
pub struct WorkerServiceIdentity {
  priv model_identity : @spec.ModelIdentity
  priv model_generation_value : UInt64
  priv bootstrap_source_digest : @worker_wire.WorkerBootstrapSourceDigest
  priv worker_limits : @worker_protocol.WorkerProtocolLimits
  priv inference_limits : @inference.InferenceLimits
} derive(Debug, Eq)

///|
pub fn WorkerServiceBinding::new(
  bootstrap_digest~ : @worker_wire.WorkerBootstrapDigest,
  bootstrap_source_digest~ : @worker_wire.WorkerBootstrapSourceDigest,
  device_ordinal~ : Int,
  inference_limits~ : @inference.InferenceLimits,
  luna_authorization? : @luna_artifact_admission.LunaAotKernelAdmission? = None,
  fp8_authorization? : @fp8_release_authority.Fp8ReusablePagedEnvelopeAuthorityV3? = None,
) -> WorkerServiceBinding raise WorkerServiceError {
  if device_ordinal < 0 ||
    device_ordinal > 1023 ||
    (luna_authorization is Some(_) && fp8_authorization is Some(_)) {
    raise Invalid(DeviceAssignment)
  }
  {
    bootstrap_digest,
    bootstrap_source_digest,
    device_ordinal,
    inference_limits,
    parent_approval: match (luna_authorization, fp8_authorization) {
      (None, None) => None
      (Some(value), None) =>
        Some(
          @worker_wire.WorkerParentApprovalSubjects::from_luna_authorization(
            value,
          ),
        )
      (None, Some(value)) =>
        Some(
          @worker_wire.WorkerParentApprovalSubjects::from_fp8_authorization(
            value,
          ),
        )
      _ => abort("worker approval constructor invariant")
    },
  }
}

///|
pub fn WorkerServiceBinding::parent_approval(
  self : WorkerServiceBinding,
) -> @worker_wire.WorkerParentApprovalSubjects? {
  self.parent_approval
}

///|
pub fn WorkerServiceBinding::bootstrap_source_digest(
  self : WorkerServiceBinding,
) -> @worker_wire.WorkerBootstrapSourceDigest {
  self.bootstrap_source_digest
}

///|
pub fn WorkerServiceBinding::bootstrap_digest(
  self : WorkerServiceBinding,
) -> @worker_wire.WorkerBootstrapDigest {
  self.bootstrap_digest
}

///|
pub fn WorkerServiceBinding::device_ordinal(self : WorkerServiceBinding) -> Int {
  self.device_ordinal
}

///|
pub fn WorkerServiceBinding::inference_limits(
  self : WorkerServiceBinding,
) -> @inference.InferenceLimits {
  self.inference_limits
}

///|
fn worker_service_identity(
  model_identity : @spec.ModelIdentity,
  model_generation_value : UInt64,
  bootstrap_source_digest : @worker_wire.WorkerBootstrapSourceDigest,
  worker_limits : @worker_protocol.WorkerProtocolLimits,
  inference_limits : @inference.InferenceLimits,
) -> WorkerServiceIdentity raise WorkerServiceError {
  if model_generation_value == 0UL {
    raise Invalid(Generation)
  }
  {
    model_identity,
    model_generation_value,
    bootstrap_source_digest,
    worker_limits,
    inference_limits,
  }
}

///|
pub fn WorkerServiceIdentity::model_identity(
  self : WorkerServiceIdentity,
) -> @spec.ModelIdentity {
  self.model_identity
}

///|
pub fn WorkerServiceIdentity::model_generation_value(
  self : WorkerServiceIdentity,
) -> UInt64 {
  self.model_generation_value
}

///|
pub fn WorkerServiceIdentity::bootstrap_source_digest(
  self : WorkerServiceIdentity,
) -> @worker_wire.WorkerBootstrapSourceDigest {
  self.bootstrap_source_digest
}

///|
pub fn WorkerServiceIdentity::worker_limits(
  self : WorkerServiceIdentity,
) -> @worker_protocol.WorkerProtocolLimits {
  self.worker_limits
}

///|
pub fn WorkerServiceIdentity::inference_limits(
  self : WorkerServiceIdentity,
) -> @inference.InferenceLimits {
  self.inference_limits
}
