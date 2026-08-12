///|
/// Untrusted aliased in-memory input for one precompiled CUDA module.
///
/// Construction performs no admission and has no path, executable metadata,
/// compiler identity, or JIT option. `admit` validates the entire set before
/// any input module is defensively copied into a returned bundle.
pub struct KernelArtifactInput {
  digest : @catalog.AotArtifactDigest
  module_bytes : FixedArray[Byte]
  function_symbol : String
}

///|
/// One admitted immutable AOT module and its single CUDA entry symbol.
///
/// The module bytes hash exactly to `digest`. This type describes inert startup
/// data only; it does not load a module, resolve a device function, or launch
/// work.
pub struct KernelArtifactEntry {
  digest : @catalog.AotArtifactDigest
  module_bytes : Bytes
  function_symbol : String
}

///|
/// Complete immutable AOT inputs required by one resolved kernel catalog.
///
/// Entries are unique and ordered by the first resolved operation that uses
/// each digest. Vendor implementations have no entry.
pub struct KernelArtifactBundle {
  model_identity : @spec.ModelIdentity
  device_target : @catalog.DeviceTarget
  catalog_version : @catalog.CatalogVersion
  entries : ReadOnlyArray[KernelArtifactEntry]
  total_module_bytes : Int64
}

///|
/// Constructs an untrusted artifact input without copying its aliased bytes.
/// Validation is deliberately centralized in `admit`, where aggregate limits
/// and catalog completeness are available.
pub fn KernelArtifactInput::new(
  digest~ : @catalog.AotArtifactDigest,
  module_bytes~ : FixedArray[Byte],
  function_symbol~ : String,
) -> KernelArtifactInput {
  { digest, module_bytes, function_symbol }
}

///|
pub fn KernelArtifactEntry::digest(
  self : KernelArtifactEntry,
) -> @catalog.AotArtifactDigest {
  self.digest
}

///|
pub fn KernelArtifactEntry::module_bytes(self : KernelArtifactEntry) -> Bytes {
  self.module_bytes
}

///|
pub fn KernelArtifactEntry::function_symbol(
  self : KernelArtifactEntry,
) -> String {
  self.function_symbol
}

///|
pub fn KernelArtifactBundle::entries(
  self : KernelArtifactBundle,
) -> ReadOnlyArray[KernelArtifactEntry] {
  self.entries
}

///|
pub fn KernelArtifactBundle::model_identity(
  self : KernelArtifactBundle,
) -> @spec.ModelIdentity {
  self.model_identity
}

///|
pub fn KernelArtifactBundle::device_target(
  self : KernelArtifactBundle,
) -> @catalog.DeviceTarget {
  self.device_target
}

///|
pub fn KernelArtifactBundle::catalog_version(
  self : KernelArtifactBundle,
) -> @catalog.CatalogVersion {
  self.catalog_version
}

///|
/// Finds one admitted entry by its approved digest without allocating.
pub fn KernelArtifactBundle::entry(
  self : KernelArtifactBundle,
  digest : @catalog.AotArtifactDigest,
) -> KernelArtifactEntry? {
  for entry in self.entries {
    if entry.digest == digest {
      return Some(entry)
    }
  }
  None
}

///|
pub fn KernelArtifactBundle::length(self : KernelArtifactBundle) -> Int {
  self.entries.length()
}

///|
pub fn KernelArtifactBundle::total_module_bytes(
  self : KernelArtifactBundle,
) -> Int64 {
  self.total_module_bytes
}
