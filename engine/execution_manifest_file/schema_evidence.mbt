///|
fn validate_identity_claim(
  root : Map[String, Json],
  expected : @spec.ModelIdentity,
) -> Unit raise ExecutionManifestFileError {
  let value = require_object(root, "model_identity", Root)
  validate_fields(value, ModelIdentity, ["content_sha256", "plan_sha256"])
  let content = require_string(value, "content_sha256", ModelIdentity)
  let plan = require_string(value, "plan_sha256", ModelIdentity)
  if !is_lower_sha256(content) {
    raise invalid_field(ModelIdentity, InvalidSha256)
  }
  if !is_lower_sha256(plan) {
    raise invalid_field(ModelIdentity, InvalidSha256, index=1)
  }
  if content != expected.content().as_hex() {
    raise IdentityMismatch(ContentDigest)
  }
  if plan != expected.plan().as_hex() {
    raise IdentityMismatch(PlanDigest)
  }
}

///|
fn validate_target_claim(
  root : Map[String, Json],
  expected : @catalog.DeviceTarget,
) -> Unit raise ExecutionManifestFileError {
  let value = require_object(root, "device_target", Root)
  validate_fields(value, DeviceTarget, [
    "compute_major", "compute_minor", "supports_bf16", "supports_cublas_lt",
  ])
  if require_nonnegative_int(value, "compute_major", DeviceTarget) !=
    expected.compute_major() {
    raise IdentityMismatch(ComputeMajor)
  }
  if require_nonnegative_int(value, "compute_minor", DeviceTarget) !=
    expected.compute_minor() {
    raise IdentityMismatch(ComputeMinor)
  }
  if require_bool(value, "supports_bf16", DeviceTarget) !=
    expected.supports_bf16() {
    raise IdentityMismatch(Bf16Support)
  }
  if require_bool(value, "supports_cublas_lt", DeviceTarget) !=
    expected.supports_cublas_lt() {
    raise IdentityMismatch(CublasLtSupport)
  }
}

///|
fn parse_profile_claim(
  root : Map[String, Json],
) -> @launch_contract.KernelProfileId raise ExecutionManifestFileError {
  let value = require_object(root, "profile", Root)
  validate_fields(value, Profile, ["id"])
  @launch_contract.KernelProfileId::from_stable_int(
    require_nonnegative_int(value, "id", Profile),
  ) catch {
    _ => raise invalid_field(Profile, UnsupportedValue)
  }
}
