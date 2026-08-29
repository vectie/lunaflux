///|
pub struct LunaKernelRootLimits {
  priv max_files : Int
  priv max_inventory_bytes : Int
  priv max_evidence_bytes : Int
}

///|
pub fn LunaKernelRootLimits::new(
  max_files~ : Int,
  max_inventory_bytes~ : Int,
  max_evidence_bytes~ : Int,
) -> LunaKernelRootLimits raise LunaKernelRootError {
  if max_files < 2 || max_inventory_bytes <= 0 || max_evidence_bytes <= 0 {
    raise Invalid(Limits)
  }
  { max_files, max_inventory_bytes, max_evidence_bytes }
}

///|
/// One immutable deployment-kernel-root payload. Paths are fixed by the typed
/// producer and cannot be supplied by a filesystem caller.
pub struct LunaKernelRootFile {
  priv relative_path : String
  priv digest_sha256 : String
  priv bytes : Bytes
}

///|
pub fn LunaKernelRootFile::relative_path(self : LunaKernelRootFile) -> String {
  self.relative_path
}

///|
pub fn LunaKernelRootFile::digest_sha256(self : LunaKernelRootFile) -> String {
  self.digest_sha256
}

///|
pub fn LunaKernelRootFile::bytes(self : LunaKernelRootFile) -> Bytes {
  self.bytes
}

///|
/// Exact root-free materialization plan consumed by the no-overwrite shell
/// assembler. `files` contains only one JSON manifest and CUBIN modules.
pub struct LunaKernelRootPlan {
  priv manifest_relative : String
  priv manifest_sha256 : String
  priv files : ReadOnlyArray[LunaKernelRootFile]
  priv inventory : Bytes
  priv inventory_sha256 : String
  priv evidence : Bytes
  priv evidence_sha256 : String
  priv identity_sha256 : String
}

///|
pub fn LunaKernelRootPlan::manifest_relative(
  self : LunaKernelRootPlan,
) -> String {
  self.manifest_relative
}

///|
pub fn LunaKernelRootPlan::manifest_sha256(self : LunaKernelRootPlan) -> String {
  self.manifest_sha256
}

///|
pub fn LunaKernelRootPlan::files(
  self : LunaKernelRootPlan,
) -> ReadOnlyArray[LunaKernelRootFile] {
  self.files
}

///|
pub fn LunaKernelRootPlan::inventory_bytes(self : LunaKernelRootPlan) -> Bytes {
  self.inventory
}

///|
pub fn LunaKernelRootPlan::inventory_sha256(
  self : LunaKernelRootPlan,
) -> String {
  self.inventory_sha256
}

///|
pub fn LunaKernelRootPlan::evidence_bytes(self : LunaKernelRootPlan) -> Bytes {
  self.evidence
}

///|
pub fn LunaKernelRootPlan::evidence_sha256(self : LunaKernelRootPlan) -> String {
  self.evidence_sha256
}

///|
pub fn LunaKernelRootPlan::identity_sha256(self : LunaKernelRootPlan) -> String {
  self.identity_sha256
}
