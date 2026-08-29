///|
pub fn TensorParallelOperand::role(
  self : TensorParallelOperand,
) -> TensorParallelOperandRole {
  self.role
}

///|
pub fn TensorParallelOperand::byte_count(self : TensorParallelOperand) -> Int64 {
  self.byte_count
}

///|
pub fn TensorParallelOperand::alignment(self : TensorParallelOperand) -> Int64 {
  self.alignment
}

///|
pub fn TensorParallelAotLaunchContract::profile(
  self : TensorParallelAotLaunchContract,
) -> @launch_contract.PagedKernelProfile {
  self.profile
}

///|
pub fn TensorParallelAotLaunchContract::operation_id(
  self : TensorParallelAotLaunchContract,
) -> @plan.OperationId {
  self.operation_id
}

///|
pub fn TensorParallelAotLaunchContract::operation_kind(
  self : TensorParallelAotLaunchContract,
) -> @plan.OperationKind {
  self.operation_kind
}

///|
pub fn TensorParallelAotLaunchContract::capability(
  self : TensorParallelAotLaunchContract,
) -> @plan.KernelCapabilityId {
  self.capability
}

///|
pub fn TensorParallelAotLaunchContract::entry_point(
  self : TensorParallelAotLaunchContract,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn TensorParallelAotLaunchContract::dimensions(
  self : TensorParallelAotLaunchContract,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn TensorParallelAotLaunchContract::input_width(
  self : TensorParallelAotLaunchContract,
) -> Int {
  self.input_width
}

///|
pub fn TensorParallelAotLaunchContract::output_width(
  self : TensorParallelAotLaunchContract,
) -> Int {
  self.output_width
}

///|
pub fn TensorParallelAotLaunchContract::inner_width(
  self : TensorParallelAotLaunchContract,
) -> Int {
  self.inner_width
}

///|
pub fn TensorParallelAotLaunchContract::query_head_count(
  self : TensorParallelAotLaunchContract,
) -> Int {
  self.query_head_count
}

///|
pub fn TensorParallelAotLaunchContract::kv_head_count(
  self : TensorParallelAotLaunchContract,
) -> Int {
  self.kv_head_count
}

///|
pub fn TensorParallelAotLaunchContract::head_dimension(
  self : TensorParallelAotLaunchContract,
) -> Int {
  self.head_dimension
}

///|
pub fn TensorParallelAotLaunchContract::operands(
  self : TensorParallelAotLaunchContract,
) -> ReadOnlyArray[TensorParallelOperand] {
  self.operands
}

///|
pub fn TensorParallelAotLaunchContract::collective_after(
  self : TensorParallelAotLaunchContract,
) -> TensorParallelCollectiveAfter {
  self.collective_after
}

///|
pub fn TensorParallelAotLaunchContractSet::model_identity(
  self : TensorParallelAotLaunchContractSet,
) -> @spec.ModelIdentity {
  self.model_identity
}

///|
pub fn TensorParallelAotLaunchContractSet::rank(
  self : TensorParallelAotLaunchContractSet,
) -> Int {
  self.rank
}

///|
pub fn TensorParallelAotLaunchContractSet::world_size(
  self : TensorParallelAotLaunchContractSet,
) -> Int {
  self.world_size
}

///|
pub fn TensorParallelAotLaunchContractSet::device_target(
  self : TensorParallelAotLaunchContractSet,
) -> @catalog.DeviceTarget {
  self.device_target
}

///|
pub fn TensorParallelAotLaunchContractSet::catalog_version(
  self : TensorParallelAotLaunchContractSet,
) -> @catalog.CatalogVersion {
  self.catalog_version
}

///|
pub fn TensorParallelAotLaunchContractSet::device_kv_layout(
  self : TensorParallelAotLaunchContractSet,
) -> @device_layout.DeviceKvLayout {
  self.device_kv_layout
}

///|
pub fn TensorParallelAotLaunchContractSet::rank_constants(
  self : TensorParallelAotLaunchContractSet,
) -> TensorParallelRankConstants {
  self.rank_constants
}

///|
pub fn TensorParallelAotLaunchContractSet::profiles(
  self : TensorParallelAotLaunchContractSet,
) -> ReadOnlyArray[@launch_contract.PagedKernelProfile] {
  self.profiles
}

///|
pub fn TensorParallelAotLaunchContractSet::contracts(
  self : TensorParallelAotLaunchContractSet,
) -> ReadOnlyArray[TensorParallelAotLaunchContract] {
  self.contracts
}
