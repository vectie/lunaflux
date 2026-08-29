///|
pub fn FusedParallelCompiledArtifactBinding::family(
  self : FusedParallelCompiledArtifactBinding,
) -> FusedParallelFamily {
  self.family
}

///|
pub fn FusedParallelCompiledArtifactBinding::candidate_digest(
  self : FusedParallelCompiledArtifactBinding,
) -> FusedParallelDigest {
  self.candidate_digest
}

///|
pub fn FusedParallelCompiledArtifactBinding::source_digest(
  self : FusedParallelCompiledArtifactBinding,
) -> FusedParallelDigest {
  self.source_digest
}

///|
pub fn FusedParallelCompiledArtifactBinding::recipe_digest(
  self : FusedParallelCompiledArtifactBinding,
) -> FusedParallelDigest {
  self.recipe_digest
}

///|
pub fn FusedParallelCompiledArtifactBinding::toolchain_digest_sha256(
  self : FusedParallelCompiledArtifactBinding,
) -> String {
  self.toolchain_digest_sha256
}

///|
pub fn FusedParallelCompiledArtifactBinding::driver_identity_sha256(
  self : FusedParallelCompiledArtifactBinding,
) -> String {
  self.driver_identity_sha256
}

///|
pub fn FusedParallelCompiledArtifactBinding::target(
  self : FusedParallelCompiledArtifactBinding,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn FusedParallelCompiledArtifactBinding::launch_dimensions(
  self : FusedParallelCompiledArtifactBinding,
) -> @launch_contract.AotLaunchDimensions {
  self.launch_dimensions
}

///|
pub fn FusedParallelCompiledArtifactBinding::function_symbol(
  self : FusedParallelCompiledArtifactBinding,
) -> String {
  self.function_symbol
}

///|
pub fn FusedParallelCompiledArtifactBinding::kernel_abi(
  self : FusedParallelCompiledArtifactBinding,
) -> FusedParallelKernelAbi {
  self.kernel_abi
}

///|
pub fn FusedParallelCompiledArtifactBinding::artifact_purpose(
  self : FusedParallelCompiledArtifactBinding,
) -> FusedParallelArtifactPurpose {
  self.purpose
}

///|
pub fn FusedParallelCompiledArtifactBinding::dispatch_canary_per_token(
  self : FusedParallelCompiledArtifactBinding,
) -> UInt {
  self.dispatch_canary_per_token
}

///|
pub fn FusedParallelCompiledArtifactBinding::qualification_candidate_digest(
  self : FusedParallelCompiledArtifactBinding,
) -> FusedParallelDigest? {
  self.qualification_candidate
}

///|
pub fn FusedParallelCompiledArtifactBinding::module_bytes(
  self : FusedParallelCompiledArtifactBinding,
) -> Bytes {
  self.module_bytes
}

///|
pub fn FusedParallelCompiledArtifactBinding::module_digest(
  self : FusedParallelCompiledArtifactBinding,
) -> FusedParallelDigest {
  self.module_digest
}

///|
pub fn FusedParallelCompiledArtifactBinding::compile_receipt_bytes(
  self : FusedParallelCompiledArtifactBinding,
) -> Bytes {
  self.compile_receipt
}

///|
pub fn FusedParallelCompiledArtifactBinding::compile_receipt_digest(
  self : FusedParallelCompiledArtifactBinding,
) -> FusedParallelDigest {
  self.compile_receipt_digest
}

///|
pub fn FusedParallelCompiledArtifactBinding::binding_record_bytes(
  self : FusedParallelCompiledArtifactBinding,
) -> Bytes {
  self.binding_record
}

///|
pub fn FusedParallelCompiledArtifactBinding::binding_digest(
  self : FusedParallelCompiledArtifactBinding,
) -> FusedParallelDigest {
  self.binding_digest
}

///|
pub fn FusedParallelCompiledArtifactBinding::manifest_bindable(
  _self : FusedParallelCompiledArtifactBinding,
) -> Bool {
  false
}
