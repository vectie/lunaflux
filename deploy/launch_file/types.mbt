///|
/// Startup bounds for the fixed deployment launch envelope.
pub struct LaunchFileLimits {
  priv maximum_bytes : Int64
  priv maximum_json_depth : Int
}

///|
/// Independent lowercase SHA-256 of exact launch-file bytes.
pub struct LaunchFileDigest(String) derive(Debug, Eq)

///|
/// Opaque external Luna deployment approval claims.
/// For tensor-parallel v8, both the capability-manifest and approved-source
/// identities are required downstream to equal the exact execution manifest.
pub struct LaunchLunaApproval {
  priv capability_manifest_sha256 : String
  priv key_identity_sha256 : String
  priv signature_sha256 : String
  priv approved_source_sha256 : String
}

///|
/// Closed runtime-recipe selection authenticated by the launch-file digest.
/// Legacy launch v1 always projects the legacy recipe; launch v2 must state
/// exactly one recipe and cannot be extended by an unknown string.
pub(all) enum LaunchRuntimeRecipe {
  DenseLlamaPagedAotV5
  DenseLlamaI8PagedAotV6
  DenseMistralBf16PagedAotV7
  DenseLlamaTensorParallelPagedAotApprovedV8
  DenseLlamaFp8ReusablePagedAotApprovedV9
} derive(Debug, Eq)

///|
/// Root-free, authority-free evidence from one closed launch snapshot.
pub struct LaunchFileAdmission {
  priv digest : LaunchFileDigest
  priv runtime_recipe : LaunchRuntimeRecipe
  priv model_root : String
  priv kernel_root : String
  priv policy_root : String
  priv descriptor_locator : String
  priv descriptor_sha256 : String
  priv policy_locator : String
  priv policy_sha256 : String
  priv worker_executable_path : String
  priv worker_executable_sha256 : String
  priv luna_approval : LaunchLunaApproval?
}

///|
/// Stable payload-free launch schema areas.
pub(all) enum LaunchFileArea {
  Argument
  Root
  RuntimeDescriptor
  InstancePolicy
  WorkerExecutable
  LunaApproval
} derive(Debug, Eq)

///|
/// Stable payload-free launch admission failures.
pub(all) suberror LaunchFileError {
  InvalidLimit
  InvalidArgument
  InvalidExpectedDigest
  InvalidField(LaunchFileArea)
  MissingField(LaunchFileArea)
  UnknownField(LaunchFileArea)
  DuplicateField
  MalformedJson
  JsonDepthExceeded
  FileUnavailable
  UnsupportedFileType
  FileTooLarge
  FileTruncated
  FileChangedDuringRead
  FileBusy
  FileOperationFailed
  DigestMismatch
  InvalidUtf8
} derive(Debug, Eq)
