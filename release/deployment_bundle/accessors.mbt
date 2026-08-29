///|
pub fn DeploymentBundleEvidence::digest(
  self : DeploymentBundleEvidence,
) -> DeploymentBundleDigest {
  self.digest
}

///|
pub fn DeploymentBundleEvidence::base_image(
  self : DeploymentBundleEvidence,
) -> String {
  self.base_image
}

///|
pub fn DeploymentBundleEvidence::architecture(
  self : DeploymentBundleEvidence,
) -> DeploymentBundleArchitecture {
  self.architecture
}

///|
pub fn DeploymentBundleEvidence::launch_file_sha256(
  self : DeploymentBundleEvidence,
) -> String {
  self.launch_file_sha256
}

///|
pub fn DeploymentBundleEvidence::runtime_descriptor_relative(
  self : DeploymentBundleEvidence,
) -> String {
  self.runtime_descriptor_relative
}

///|
pub fn DeploymentBundleEvidence::runtime_descriptor_sha256(
  self : DeploymentBundleEvidence,
) -> String {
  self.runtime_descriptor_sha256
}

///|
pub fn DeploymentBundleEvidence::instance_policy_relative(
  self : DeploymentBundleEvidence,
) -> String {
  self.instance_policy_relative
}

///|
pub fn DeploymentBundleEvidence::instance_policy_sha256(
  self : DeploymentBundleEvidence,
) -> String {
  self.instance_policy_sha256
}

///|
pub fn DeploymentBundleEvidence::lunaflux_executable_sha256(
  self : DeploymentBundleEvidence,
) -> String {
  self.lunaflux_executable_sha256
}

///|
pub fn DeploymentBundleEvidence::worker_executable_sha256(
  self : DeploymentBundleEvidence,
) -> String {
  self.worker_executable_sha256
}

///|
pub fn DeploymentBundleEvidence::model_inventory_sha256(
  self : DeploymentBundleEvidence,
) -> String {
  self.model_inventory_sha256
}

///|
pub fn DeploymentBundleEvidence::policy_inventory_sha256(
  self : DeploymentBundleEvidence,
) -> String {
  self.policy_inventory_sha256
}

///|
pub fn DeploymentBundleEvidence::kernel_manifest_relative(
  self : DeploymentBundleEvidence,
) -> String {
  self.kernel_manifest_relative
}

///|
pub fn DeploymentBundleEvidence::kernel_manifest_sha256(
  self : DeploymentBundleEvidence,
) -> String {
  self.kernel_manifest_sha256
}

///|
pub fn DeploymentBundleEvidence::kernel_inventory_sha256(
  self : DeploymentBundleEvidence,
) -> String {
  self.kernel_inventory_sha256
}

///|
pub fn DeploymentBundleEvidence::runtime_library_inventory_sha256(
  self : DeploymentBundleEvidence,
) -> String {
  self.runtime_library_inventory_sha256
}

///|
pub fn DeploymentBundleEvidence::rootfs_inventory_sha256(
  self : DeploymentBundleEvidence,
) -> String {
  self.rootfs_inventory_sha256
}

///|
pub fn DeploymentBundleEvidence::assembly_input_sha256(
  self : DeploymentBundleEvidence,
) -> String {
  self.assembly_input_sha256
}

///|
pub fn DeploymentBundleEvidence::assembler_sha256(
  self : DeploymentBundleEvidence,
) -> String {
  self.assembler_sha256
}

///|
pub fn DeploymentBundleEvidence::canonical_bytes(
  self : DeploymentBundleEvidence,
) -> Bytes {
  self.canonical
}
