///|
/// Fields accepted by the model configuration constructor.
pub(all) enum ModelField {
  ArtifactPath
  ExpectedDigest
  ArchitectureAllowlist
  ArchitectureName
} derive(Debug, Eq)

///|
/// Bounded validation categories for model settings.
pub(all) enum ModelIssue {
  Empty
  TooLong
  TooMany
  InvalidFormat
  Duplicate
} derive(Debug, Eq)

///|
/// Payload-safe model configuration validation failure.
pub(all) suberror ModelConfigError {
  Invalid(field~ : ModelField, issue~ : ModelIssue, index~ : Int)
} derive(Debug, Eq)

///|
/// Immutable artifact identity and architecture policy.
pub struct ModelConfig {
  artifact_path : String
  expected_digest : String
  architecture_allowlist : ReadOnlyArray[String]
} derive(Debug, Eq)

///|
/// Constructs validated model settings.
///
/// The expected digest uses the canonical `sha256:` prefix followed by exactly
/// 64 lowercase hexadecimal digits. Architecture names are deliberately a
/// small, safe vocabulary suitable for diagnostics and manifest matching.
pub fn new(
  artifact_path : String,
  expected_digest : String,
  architecture_allowlist : ArrayView[String],
) -> ModelConfig raise ModelConfigError {
  if artifact_path.length() == 0 {
    raise Invalid(field=ArtifactPath, issue=Empty, index=0)
  }
  if artifact_path.length() > 4096 {
    raise Invalid(field=ArtifactPath, issue=TooLong, index=0)
  }
  if !valid_sha256(expected_digest) {
    raise Invalid(field=ExpectedDigest, issue=InvalidFormat, index=0)
  }
  if architecture_allowlist.length() == 0 {
    raise Invalid(field=ArchitectureAllowlist, issue=Empty, index=0)
  }
  if architecture_allowlist.length() > 16 {
    raise Invalid(field=ArchitectureAllowlist, issue=TooMany, index=16)
  }
  for index, name in architecture_allowlist {
    if name.length() == 0 {
      raise Invalid(field=ArchitectureName, issue=Empty, index~)
    }
    if name.length() > 64 {
      raise Invalid(field=ArchitectureName, issue=TooLong, index~)
    }
    if !valid_architecture_name(name) {
      raise Invalid(field=ArchitectureName, issue=InvalidFormat, index~)
    }
    for prior in 0..<index {
      if architecture_allowlist[prior] == name {
        raise Invalid(field=ArchitectureName, issue=Duplicate, index~)
      }
    }
  }
  {
    artifact_path,
    expected_digest,
    architecture_allowlist: ReadOnlyArray::from_array(architecture_allowlist),
  }
}

///|
fn valid_sha256(digest : StringView) -> Bool {
  if digest.length() != 71 || !digest.has_prefix("sha256:") {
    return false
  }
  for index in 7..<71 {
    let unit = digest[index]
    if !((unit >= '0' && unit <= '9') || (unit >= 'a' && unit <= 'f')) {
      return false
    }
  }
  true
}

///|
fn valid_architecture_name(name : StringView) -> Bool {
  for char in name {
    if !((char >= 'a' && char <= 'z') ||
      (char >= 'A' && char <= 'Z') ||
      (char >= '0' && char <= '9') ||
      char == '-' ||
      char == '_' ||
      char == '.') {
      return false
    }
  }
  true
}

///|
pub fn ModelConfig::artifact_path(self : ModelConfig) -> String {
  self.artifact_path
}

///|
pub fn ModelConfig::expected_digest(self : ModelConfig) -> String {
  self.expected_digest
}

///|
pub fn ModelConfig::architecture_allowlist(
  self : ModelConfig,
) -> ReadOnlyArray[String] {
  self.architecture_allowlist
}
