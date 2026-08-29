///|
pub(all) enum KernelConfigField {
  ManifestLocator
  ExpectedManifestDigest
  AdmissionPolicy
} derive(Debug, Eq)

///|
pub(all) enum KernelConfigIssue {
  InvalidFormat
  Unsupported
} derive(Debug, Eq)

///|
pub(all) suberror KernelConfigError {
  Invalid(field~ : KernelConfigField, issue~ : KernelConfigIssue)
} derive(Debug, Eq)

///|
pub(all) enum KernelAdmissionPolicy {
  DeploymentApprovedAotOnly
  OfflineInspectionOnly
} derive(Debug, Eq)

///|
pub struct KernelManifestDigest(String) derive(Debug, Eq, Hash)

///|
pub fn KernelManifestDigest::from_sha256(
  value : StringView,
) -> KernelManifestDigest raise KernelConfigError {
  if value.length() != 64 {
    raise Invalid(field=ExpectedManifestDigest, issue=InvalidFormat)
  }
  for unit in value {
    if !(unit is ('0'..='9' | 'a'..='f')) {
      raise Invalid(field=ExpectedManifestDigest, issue=InvalidFormat)
    }
  }
  KernelManifestDigest(value.to_owned())
}

///|
pub fn KernelManifestDigest::as_hex(self : KernelManifestDigest) -> String {
  self.0
}

///|
pub struct KernelConfig {
  priv manifest_locator : @approved_fs.ApprovedRelativeLocator
  priv expected_manifest_digest : KernelManifestDigest
  priv admission_policy : KernelAdmissionPolicy
}

///|
/// Admits the only kernel configuration authority needed by runtime startup.
/// The locator is a strict relative descendant, the digest is exact lowercase
/// SHA-256, and policy cannot enable source compilation or runtime JIT.
pub fn new(
  manifest_locator : StringView,
  expected_manifest_digest : StringView,
  admission_policy : KernelAdmissionPolicy,
) -> KernelConfig raise KernelConfigError {
  let locator = @approved_fs.ApprovedRelativeLocator::new(manifest_locator) catch {
    _ => raise Invalid(field=ManifestLocator, issue=InvalidFormat)
  }
  let digest = KernelManifestDigest::from_sha256(expected_manifest_digest)
  {
    manifest_locator: locator,
    expected_manifest_digest: digest,
    admission_policy,
  }
}

///|
pub fn KernelConfig::manifest_locator(
  self : KernelConfig,
) -> @approved_fs.ApprovedRelativeLocator {
  self.manifest_locator
}

///|
pub fn KernelConfig::expected_manifest_digest(
  self : KernelConfig,
) -> KernelManifestDigest {
  self.expected_manifest_digest
}

///|
pub fn KernelConfig::admission_policy(
  self : KernelConfig,
) -> KernelAdmissionPolicy {
  self.admission_policy
}

///|
pub fn KernelConfig::permits_runtime_readiness(self : KernelConfig) -> Bool {
  self.admission_policy == DeploymentApprovedAotOnly
}
