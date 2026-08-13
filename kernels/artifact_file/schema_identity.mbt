///|
fn validate_identity(
  root : Map[String, Json],
  expected : @spec.ModelIdentity,
) -> Unit raise KernelArtifactFileError {
  let identity = require_object(
    root,
    "model_identity",
    ManifestRoot,
    ModelIdentityField,
  )
  validate_fields(identity, ManifestModelIdentity, [
    ("content_sha256", ContentSha256Field),
    ("plan_sha256", PlanSha256Field),
  ])
  let content = require_string(
    identity,
    "content_sha256",
    ManifestModelIdentity,
    ContentSha256Field,
  )
  let plan = require_string(
    identity,
    "plan_sha256",
    ManifestModelIdentity,
    PlanSha256Field,
  )
  if !is_lower_sha256(content) {
    raise invalid_field(
      ManifestModelIdentity,
      ContentSha256Field,
      InvalidSha256,
    )
  }
  if !is_lower_sha256(plan) {
    raise invalid_field(ManifestModelIdentity, PlanSha256Field, InvalidSha256)
  }
  if content != expected.content().as_hex() {
    raise IdentityMismatch(ModelContentIdentity)
  }
  if plan != expected.plan().as_hex() {
    raise IdentityMismatch(ModelPlanIdentity)
  }
}

///|
fn validate_device_target(
  root : Map[String, Json],
  expected : @catalog.DeviceTarget,
) -> Unit raise KernelArtifactFileError {
  let target = require_object(
    root,
    "device_target",
    ManifestRoot,
    DeviceTargetField,
  )
  validate_fields(target, ManifestDeviceTarget, [
    ("compute_major", ComputeMajorField),
    ("compute_minor", ComputeMinorField),
    ("supports_bf16", SupportsBf16Field),
    ("supports_cublas_lt", SupportsCublasLtField),
  ])
  if require_nonnegative_int(
      target,
      "compute_major",
      ManifestDeviceTarget,
      ComputeMajorField,
    ) !=
    expected.compute_major() {
    raise IdentityMismatch(DeviceComputeMajor)
  }
  if require_nonnegative_int(
      target,
      "compute_minor",
      ManifestDeviceTarget,
      ComputeMinorField,
    ) !=
    expected.compute_minor() {
    raise IdentityMismatch(DeviceComputeMinor)
  }
  if require_bool(
      target,
      "supports_bf16",
      ManifestDeviceTarget,
      SupportsBf16Field,
    ) !=
    expected.supports_bf16() {
    raise IdentityMismatch(DeviceBf16Support)
  }
  if require_bool(
      target,
      "supports_cublas_lt",
      ManifestDeviceTarget,
      SupportsCublasLtField,
    ) !=
    expected.supports_cublas_lt() {
    raise IdentityMismatch(DeviceCublasLtSupport)
  }
}

///|
fn validate_catalog_version(
  root : Map[String, Json],
  expected : @catalog.CatalogVersion,
) -> Unit raise KernelArtifactFileError {
  let value = require_nonnegative_int(
    root,
    "catalog_version",
    ManifestRoot,
    CatalogVersionField,
  )
  if value != expected.as_int() {
    raise IdentityMismatch(KernelCatalogVersion)
  }
}

///|
fn stateless_contract_evidence(
  contracts : @launch_contract.LaunchContractSet,
) -> ManifestContractEvidence {
  {
    model_identity: contracts.model_identity(),
    device_target: contracts.device_target(),
    catalog_version: contracts.catalog_version(),
  }
}

///|
fn paged_contract_evidence(
  contracts : @launch_contract.PagedKvLaunchContractSet,
) -> ManifestContractEvidence {
  {
    model_identity: contracts.model_identity(),
    device_target: contracts.device_target(),
    catalog_version: contracts.catalog_version(),
  }
}
