///|
pub fn LaunchFileAdmission::digest(
  self : LaunchFileAdmission,
) -> LaunchFileDigest {
  self.digest
}

///|
pub fn LaunchFileAdmission::runtime_recipe(
  self : LaunchFileAdmission,
) -> LaunchRuntimeRecipe {
  self.runtime_recipe
}

///|
pub fn LaunchFileAdmission::model_root_label(
  self : LaunchFileAdmission,
) -> String {
  self.model_root
}

///|
pub fn LaunchFileAdmission::kernel_root_label(
  self : LaunchFileAdmission,
) -> String {
  self.kernel_root
}

///|
pub fn LaunchFileAdmission::policy_root_label(
  self : LaunchFileAdmission,
) -> String {
  self.policy_root
}

///|
pub fn LaunchFileAdmission::descriptor_locator(
  self : LaunchFileAdmission,
) -> String {
  self.descriptor_locator
}

///|
pub fn LaunchFileAdmission::descriptor_sha256(
  self : LaunchFileAdmission,
) -> String {
  self.descriptor_sha256
}

///|
pub fn LaunchFileAdmission::policy_locator(
  self : LaunchFileAdmission,
) -> String {
  self.policy_locator
}

///|
pub fn LaunchFileAdmission::policy_sha256(self : LaunchFileAdmission) -> String {
  self.policy_sha256
}

///|
pub fn LaunchFileAdmission::worker_executable_path(
  self : LaunchFileAdmission,
) -> String {
  self.worker_executable_path
}

///|
pub fn LaunchFileAdmission::worker_executable_sha256(
  self : LaunchFileAdmission,
) -> String {
  self.worker_executable_sha256
}

///|
pub fn LaunchFileAdmission::luna_approval(
  self : LaunchFileAdmission,
) -> LaunchLunaApproval? {
  self.luna_approval
}

///|
pub fn LaunchLunaApproval::capability_manifest_sha256(
  self : LaunchLunaApproval,
) -> String {
  self.capability_manifest_sha256
}

///|
pub fn LaunchLunaApproval::key_identity_sha256(
  self : LaunchLunaApproval,
) -> String {
  self.key_identity_sha256
}

///|
pub fn LaunchLunaApproval::signature_sha256(
  self : LaunchLunaApproval,
) -> String {
  self.signature_sha256
}

///|
pub fn LaunchLunaApproval::approved_source_sha256(
  self : LaunchLunaApproval,
) -> String {
  self.approved_source_sha256
}
