///|
/// Explicit SHA-256 identity supplied independently of the manifest file.
pub struct KernelManifestDigest {
  priv value : String
} derive(Debug, Eq)

///|
/// Untrusted pairing of one expected module digest and relative file locator.
///
/// Construction retains an already immutable locator without copying its
/// bytes. `KernelArtifactManifest::admit_files` applies the caller's exact path
/// and collection bounds before this input can enter an admitted source.
pub struct KernelModuleFileInput {
  priv digest : @catalog.AotArtifactDigest
  priv path_identifier : String
}

///|
/// Parsed or directly admitted inert AOT module-file source.
///
/// Its arrays and path-bearing values remain private. The value grants no
/// filesystem authority and owns no file handle or module bytes.
pub struct KernelArtifactManifest {
  priv module_digests : ReadOnlyArray[@catalog.AotArtifactDigest]
  priv module_path_identifiers : ReadOnlyArray[String]
  priv entry_points : ReadOnlyArray[@artifact.KernelEntryPointInput]
}

///|
priv struct ManifestContractEvidence {
  model_identity : @spec.ModelIdentity
  device_target : @catalog.DeviceTarget
  catalog_version : @catalog.CatalogVersion
}

///|
priv struct LoadedArtifactInputs {
  manifest : KernelArtifactManifest
  module_inputs : Array[@artifact.KernelModuleInput]
}

///|
fn is_lower_sha256(value : StringView) -> Bool {
  if value.length() != 64 {
    return false
  }
  for unit in value {
    if !(unit is ('0'..='9' | 'a'..='f')) {
      return false
    }
  }
  true
}

///|
pub fn KernelManifestDigest::from_sha256(
  value : String,
) -> KernelManifestDigest raise KernelArtifactFileError {
  if !is_lower_sha256(value) {
    raise InvalidExpectedManifestDigest
  }
  { value, }
}

///|
pub fn KernelManifestDigest::as_hex(self : KernelManifestDigest) -> String {
  self.value
}

///|
/// Retains one untrusted digest/locator pairing for later bounded admission.
pub fn KernelModuleFileInput::new(
  digest~ : @catalog.AotArtifactDigest,
  path_identifier~ : String,
) -> KernelModuleFileInput {
  { digest, path_identifier }
}

///|
/// Admits an inert exact module-file source from already parsed declarations.
///
/// Every locator is a strict relative identifier under the same path policy as
/// JSON manifest admission. Duplicate paths and bounded collection overflow
/// fail before immutable source ownership is published. Symbol syntax/length,
/// entry-point-to-module linkage, and module bytes deliberately remain subject
/// to downstream semantic artifact admission; `KernelEntryPointInput` exposes
/// read-only identity/symbol access but no mutation or result authority.
pub fn KernelArtifactManifest::admit_files(
  modules : ArrayView[KernelModuleFileInput],
  entry_points : ArrayView[@artifact.KernelEntryPointInput],
  limits : KernelArtifactFileLimits,
) -> KernelArtifactManifest raise KernelArtifactFileError {
  let artifact_limits = limits.artifact()
  if modules.length() > artifact_limits.max_modules() {
    raise SourceCountExceeded(
      source=SourceModules,
      maximum=artifact_limits.max_modules(),
    )
  }
  if entry_points.length() > artifact_limits.max_entry_points() {
    raise SourceCountExceeded(
      source=SourceEntryPoints,
      maximum=artifact_limits.max_entry_points(),
    )
  }
  let digests : Array[@catalog.AotArtifactDigest] = []
  let paths : Array[String] = []
  let seen_paths : Map[String, Bool] = Map([])
  for index, input in modules {
    validate_path_identifier(
      input.path_identifier,
      ModuleFile(index),
      limits.max_path_bytes(),
    )
    if seen_paths.contains(input.path_identifier) {
      raise InvalidPath(file=ModuleFile(index), issue=DuplicatePath)
    }
    seen_paths[input.path_identifier] = true
    digests.push(input.digest)
    paths.push(input.path_identifier)
  }
  {
    module_digests: ReadOnlyArray::from_array(digests),
    module_path_identifiers: ReadOnlyArray::from_array(paths),
    entry_points: ReadOnlyArray::from_array(entry_points),
  }
}

///|
pub fn KernelArtifactManifest::module_count(
  self : KernelArtifactManifest,
) -> Int {
  self.module_digests.length()
}

///|
pub fn KernelArtifactManifest::entry_point_count(
  self : KernelArtifactManifest,
) -> Int {
  self.entry_points.length()
}
