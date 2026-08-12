///|
/// Input for one precompiled content-addressed CUDA module.
///
/// The fixed-array constructor retains an untrusted alias until every small
/// semantic check passes; admission then makes exactly one defensive ownership
/// copy. The Bytes constructor accepts an already immutable owned snapshot
/// without another module-sized copy. The private variants make mixed or
/// missing ownership states unrepresentable outside this package.
pub enum KernelModuleInput {
  AliasedModule(@catalog.AotArtifactDigest, FixedArray[Byte])
  OwnedModule(@catalog.AotArtifactDigest, Bytes)
}

///|
/// Untrusted symbol mapping for one exact stable entry-point identity.
pub struct KernelEntryPointInput {
  entry_point : @catalog.AotKernelEntryPoint
  function_symbol : String
}

///|
/// One admitted immutable AOT module. Each digest is retained exactly once.
pub struct KernelModule {
  digest : @catalog.AotArtifactDigest
  module_bytes : Bytes
}

///|
/// One admitted immutable mapping from stable identity to bounded CUDA symbol.
pub struct KernelEntryPoint {
  entry_point : @catalog.AotKernelEntryPoint
  module_index : Int
  function_symbol : String
}

///|
/// Complete immutable modules and entry points required by one exact launch
/// contract set. Runtime lookup uses integer identities over ordered arrays.
pub struct KernelArtifactBundle {
  model_identity : @spec.ModelIdentity
  device_target : @catalog.DeviceTarget
  catalog_version : @catalog.CatalogVersion
  modules : ReadOnlyArray[KernelModule]
  entry_points : ReadOnlyArray[KernelEntryPoint]
  total_module_bytes : Int64
}

///|
pub fn KernelModuleInput::new(
  digest~ : @catalog.AotArtifactDigest,
  module_bytes~ : FixedArray[Byte],
) -> KernelModuleInput {
  AliasedModule(digest, module_bytes)
}

///|
/// Constructs an artifact input from an immutable exact file snapshot.
///
/// `Bytes` cannot be mutated after construction, so file admission may pass
/// ownership through this boundary without creating a second full snapshot.
pub fn KernelModuleInput::from_bytes(
  digest~ : @catalog.AotArtifactDigest,
  module_bytes~ : Bytes,
) -> KernelModuleInput {
  OwnedModule(digest, module_bytes)
}

///|
pub fn KernelEntryPointInput::new(
  entry_point~ : @catalog.AotKernelEntryPoint,
  function_symbol~ : String,
) -> KernelEntryPointInput {
  { entry_point, function_symbol }
}

///|
pub fn KernelModule::digest(self : KernelModule) -> @catalog.AotArtifactDigest {
  self.digest
}

///|
pub fn KernelModule::module_bytes(self : KernelModule) -> Bytes {
  self.module_bytes
}

///|
pub fn KernelEntryPoint::identity(
  self : KernelEntryPoint,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn KernelEntryPoint::module_index(self : KernelEntryPoint) -> Int {
  self.module_index
}

///|
pub fn KernelEntryPoint::function_symbol(self : KernelEntryPoint) -> String {
  self.function_symbol
}

///|
pub fn KernelArtifactBundle::modules(
  self : KernelArtifactBundle,
) -> ReadOnlyArray[KernelModule] {
  self.modules
}

///|
pub fn KernelArtifactBundle::entry_points(
  self : KernelArtifactBundle,
) -> ReadOnlyArray[KernelEntryPoint] {
  self.entry_points
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
pub fn KernelArtifactBundle::find_module(
  self : KernelArtifactBundle,
  digest : @catalog.AotArtifactDigest,
) -> KernelModule? {
  for item in self.modules {
    if item.digest == digest {
      return Some(item)
    }
  }
  None
}

///|
pub fn KernelArtifactBundle::entry_point(
  self : KernelArtifactBundle,
  identity : @catalog.AotKernelEntryPoint,
) -> KernelEntryPoint? {
  for entry in self.entry_points {
    if entry.entry_point == identity {
      return Some(entry)
    }
  }
  None
}

///|
pub fn KernelArtifactBundle::module_count(self : KernelArtifactBundle) -> Int {
  self.modules.length()
}

///|
pub fn KernelArtifactBundle::entry_point_count(
  self : KernelArtifactBundle,
) -> Int {
  self.entry_points.length()
}

///|
pub fn KernelArtifactBundle::total_module_bytes(
  self : KernelArtifactBundle,
) -> Int64 {
  self.total_module_bytes
}
