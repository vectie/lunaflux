///|
/// Exact supported Linux architecture of one staged deployment bundle.
pub(all) enum DeploymentBundleArchitecture {
  X86_64
  Aarch64
} derive(Debug, Eq)

///|
/// Independent lowercase SHA-256 of the canonical bundle-evidence bytes.
pub struct DeploymentBundleDigest(String) derive(Debug, Eq, Hash)

///|
/// Startup bounds for canonical bundle-evidence admission.
pub struct DeploymentBundleLimits {
  priv maximum_bytes : Int
  priv maximum_base_image_bytes : Int
  priv maximum_locator_bytes : Int
}

///|
/// Root-free evidence for one exact, compiler/JIT-free staged deployment.
///
/// The record owns no filesystem, image, model, process, or reviewer
/// authority. The release host must still run the exact bundle and OCI
/// verifiers against the staged bytes.
pub struct DeploymentBundleEvidence {
  priv digest : DeploymentBundleDigest
  priv base_image : String
  priv architecture : DeploymentBundleArchitecture
  priv launch_file_sha256 : String
  priv runtime_descriptor_relative : String
  priv runtime_descriptor_sha256 : String
  priv instance_policy_relative : String
  priv instance_policy_sha256 : String
  priv lunaflux_executable_sha256 : String
  priv worker_executable_sha256 : String
  priv model_inventory_sha256 : String
  priv policy_inventory_sha256 : String
  priv kernel_manifest_relative : String
  priv kernel_manifest_sha256 : String
  priv kernel_inventory_sha256 : String
  priv runtime_library_inventory_sha256 : String
  priv rootfs_inventory_sha256 : String
  priv assembly_input_sha256 : String
  priv assembler_sha256 : String
  priv canonical : Bytes
}

///|
/// Stable payload-free bundle-evidence areas.
pub(all) enum DeploymentBundleArea {
  Evidence
  BaseImage
  Architecture
  LaunchFile
  RuntimeDescriptor
  InstancePolicy
  Executable
  ModelInventory
  PolicyInventory
  KernelManifest
  KernelInventory
  RuntimeLibraryInventory
  RootfsInventory
  AssemblyInput
  Assembler
  Policy
} derive(Debug, Eq)

///|
/// Stable payload-free bundle-evidence admission failures.
pub(all) suberror DeploymentBundleError {
  InvalidLimit
  InvalidExpectedDigest
  EvidenceTooLarge
  InvalidUtf8
  NonCanonicalEvidence
  InvalidField(DeploymentBundleArea)
  DigestMismatch
  ReleaseEvidenceRejected
} derive(Debug, Eq)
