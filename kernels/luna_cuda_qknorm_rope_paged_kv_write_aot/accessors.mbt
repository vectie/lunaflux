///|
pub fn QkNormRopePagedKvWriteCudaCandidate::semantic_plan(
  self : QkNormRopePagedKvWriteCudaCandidate,
) -> @luna_fusion_plan.QkvQkNormPositionedRopePagedKvWritePlan {
  self.semantic
}

///|
pub fn QkNormRopePagedKvWriteCudaCandidate::target(
  self : QkNormRopePagedKvWriteCudaCandidate,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn QkNormRopePagedKvWriteCudaCandidate::profile(
  self : QkNormRopePagedKvWriteCudaCandidate,
) -> @launch_contract.PagedKernelProfile {
  self.profile
}

///|
pub fn QkNormRopePagedKvWriteCudaCandidate::compiler_policy(
  self : QkNormRopePagedKvWriteCudaCandidate,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn QkNormRopePagedKvWriteCudaCandidate::kv_layout(
  self : QkNormRopePagedKvWriteCudaCandidate,
) -> @device_layout.DeviceKvLayout {
  self.layout
}

///|
pub fn QkNormRopePagedKvWriteCudaCandidate::operands(
  self : QkNormRopePagedKvWriteCudaCandidate,
) -> ReadOnlyArray[@launch_contract.AotOperand] {
  self.operands
}

///|
pub fn QkNormRopePagedKvWriteCudaCandidate::launch_dimensions(
  self : QkNormRopePagedKvWriteCudaCandidate,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn QkNormRopePagedKvWriteCudaCandidate::function_symbol(
  self : QkNormRopePagedKvWriteCudaCandidate,
) -> String {
  self.symbol
}

///|
pub fn QkNormRopePagedKvWriteCudaCandidate::kernel_abi(
  self : QkNormRopePagedKvWriteCudaCandidate,
) -> QkNormRopePagedKvWriteKernelAbi {
  self.abi
}

///|
pub fn QkNormRopePagedKvWriteCudaCandidate::alias_contract(
  self : QkNormRopePagedKvWriteCudaCandidate,
) -> QkNormRopePagedKvWriteAliasContract {
  self.alias_contract
}

///|
pub fn QkNormRopePagedKvWriteCudaCandidate::source_bytes(
  self : QkNormRopePagedKvWriteCudaCandidate,
) -> Bytes {
  self.source
}

///|
pub fn QkNormRopePagedKvWriteCudaCandidate::source_digest(
  self : QkNormRopePagedKvWriteCudaCandidate,
) -> QkNormRopePagedKvWriteDigest {
  self.source_digest
}

///|
pub fn QkNormRopePagedKvWriteCudaCandidate::recipe_bytes(
  self : QkNormRopePagedKvWriteCudaCandidate,
) -> Bytes {
  self.recipe
}

///|
pub fn QkNormRopePagedKvWriteCudaCandidate::recipe_digest(
  self : QkNormRopePagedKvWriteCudaCandidate,
) -> QkNormRopePagedKvWriteDigest {
  self.recipe_digest
}

///|
pub fn QkNormRopePagedKvWriteCudaCandidate::candidate_digest(
  self : QkNormRopePagedKvWriteCudaCandidate,
) -> QkNormRopePagedKvWriteDigest {
  self.candidate_digest
}

///|
pub fn QkNormRopePagedKvWriteCudaCandidate::manifest_bindable(
  _self : QkNormRopePagedKvWriteCudaCandidate,
) -> Bool {
  false
}

///|
pub fn QkNormRopePagedKvWriteCompiledModule::module_bytes(
  self : QkNormRopePagedKvWriteCompiledModule,
) -> Bytes {
  self.module_bytes
}

///|
pub fn QkNormRopePagedKvWriteCompiledModule::module_digest(
  self : QkNormRopePagedKvWriteCompiledModule,
) -> QkNormRopePagedKvWriteDigest {
  self.module_digest
}

///|
pub fn QkNormRopePagedKvWriteCompiledModule::compile_receipt_bytes(
  self : QkNormRopePagedKvWriteCompiledModule,
) -> Bytes {
  self.receipt
}

///|
pub fn QkNormRopePagedKvWriteCompiledModule::compile_receipt_digest(
  self : QkNormRopePagedKvWriteCompiledModule,
) -> QkNormRopePagedKvWriteDigest {
  self.receipt_digest
}

///|
pub fn QkNormRopePagedKvWriteCudaBinding::function_symbol(
  self : QkNormRopePagedKvWriteCudaBinding,
) -> String {
  self.candidate.symbol
}

///|
pub fn QkNormRopePagedKvWriteCudaBinding::kernel_abi(
  self : QkNormRopePagedKvWriteCudaBinding,
) -> QkNormRopePagedKvWriteKernelAbi {
  self.candidate.abi
}

///|
pub fn QkNormRopePagedKvWriteCudaBinding::operands(
  self : QkNormRopePagedKvWriteCudaBinding,
) -> ReadOnlyArray[@launch_contract.AotOperand] {
  self.candidate.operands
}

///|
pub fn QkNormRopePagedKvWriteCudaBinding::launch_dimensions(
  self : QkNormRopePagedKvWriteCudaBinding,
) -> @launch_contract.AotLaunchDimensions {
  self.candidate.dimensions
}

///|
pub fn QkNormRopePagedKvWriteCudaBinding::alias_contract(
  self : QkNormRopePagedKvWriteCudaBinding,
) -> QkNormRopePagedKvWriteAliasContract {
  self.candidate.alias_contract
}

///|
pub fn QkNormRopePagedKvWriteCudaBinding::module_bytes(
  self : QkNormRopePagedKvWriteCudaBinding,
) -> Bytes {
  self.compiled.module_bytes
}

///|
pub fn QkNormRopePagedKvWriteCudaBinding::module_digest(
  self : QkNormRopePagedKvWriteCudaBinding,
) -> QkNormRopePagedKvWriteDigest {
  self.compiled.module_digest
}

///|
pub fn QkNormRopePagedKvWriteCudaBinding::binding_record_bytes(
  self : QkNormRopePagedKvWriteCudaBinding,
) -> Bytes {
  self.canonical
}

///|
pub fn QkNormRopePagedKvWriteCudaBinding::binding_digest(
  self : QkNormRopePagedKvWriteCudaBinding,
) -> QkNormRopePagedKvWriteDigest {
  self.digest
}

///|
pub fn QkNormRopePagedKvWriteCudaBinding::manifest_bindable(
  _self : QkNormRopePagedKvWriteCudaBinding,
) -> Bool {
  false
}
