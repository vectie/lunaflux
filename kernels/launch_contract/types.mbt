///|
/// Untrusted declarative input for one exact AOT launch.
///
/// Construction defensively owns the small operand table. Admission against a
/// model plan and resolved catalog remains mandatory.
pub struct AotLaunchContractInput {
  priv profile_id : KernelProfileId
  priv operation_id : @plan.OperationId
  priv entry_point : @catalog.AotKernelEntryPoint
  priv dimensions : AotLaunchDimensions
  priv operands : ReadOnlyArray[AotOperand]
} derive(Debug, Eq)

///|
/// One fully admitted, exact, allocation-free launch description.
pub struct AotLaunchContract {
  priv profile : KernelProfile
  priv operation_id : @plan.OperationId
  priv operation_kind : @plan.OperationKind
  priv operation_shape : @plan.OperationShape
  priv capability : @plan.KernelCapabilityId
  priv entry_point : @catalog.AotKernelEntryPoint
  priv dimensions : AotLaunchDimensions
  priv operands : ReadOnlyArray[AotOperand]
  priv workspace_bytes : Int64
  priv workspace_alignment : Int64
} derive(Debug, Eq)

///|
/// Immutable startup result ordered by profile and then model operation.
pub struct LaunchContractSet {
  priv model_identity : @spec.ModelIdentity
  priv device_target : @catalog.DeviceTarget
  priv catalog_version : @catalog.CatalogVersion
  priv profiles : ReadOnlyArray[KernelProfile]
  priv contracts : ReadOnlyArray[AotLaunchContract]
} derive(Debug, Eq)

///|
pub fn AotLaunchContractInput::new(
  profile_id~ : KernelProfileId,
  operation_id~ : @plan.OperationId,
  entry_point~ : @catalog.AotKernelEntryPoint,
  dimensions~ : AotLaunchDimensions,
  operands~ : ArrayView[AotOperand],
) -> AotLaunchContractInput raise LaunchContractError {
  {
    profile_id,
    operation_id,
    entry_point,
    dimensions,
    operands: checked_owned_operands(operands, operation_id),
  }
}

///|
pub fn AotLaunchContract::profile(self : AotLaunchContract) -> KernelProfile {
  self.profile
}

///|
pub fn AotLaunchContract::operation_id(
  self : AotLaunchContract,
) -> @plan.OperationId {
  self.operation_id
}

///|
pub fn AotLaunchContract::operation_kind(
  self : AotLaunchContract,
) -> @plan.OperationKind {
  self.operation_kind
}

///|
pub fn AotLaunchContract::operation_shape(
  self : AotLaunchContract,
) -> @plan.OperationShape {
  self.operation_shape
}

///|
pub fn AotLaunchContract::capability(
  self : AotLaunchContract,
) -> @plan.KernelCapabilityId {
  self.capability
}

///|
pub fn AotLaunchContract::entry_point(
  self : AotLaunchContract,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn AotLaunchContract::dimensions(
  self : AotLaunchContract,
) -> AotLaunchDimensions {
  self.dimensions
}

///|
pub fn AotLaunchContract::operands(
  self : AotLaunchContract,
) -> ReadOnlyArray[AotOperand] {
  self.operands
}

///|
pub fn AotLaunchContract::workspace_bytes(self : AotLaunchContract) -> Int64 {
  self.workspace_bytes
}

///|
pub fn AotLaunchContract::workspace_alignment(
  self : AotLaunchContract,
) -> Int64 {
  self.workspace_alignment
}

///|
pub fn LaunchContractSet::model_identity(
  self : LaunchContractSet,
) -> @spec.ModelIdentity {
  self.model_identity
}

///|
pub fn LaunchContractSet::device_target(
  self : LaunchContractSet,
) -> @catalog.DeviceTarget {
  self.device_target
}

///|
pub fn LaunchContractSet::catalog_version(
  self : LaunchContractSet,
) -> @catalog.CatalogVersion {
  self.catalog_version
}

///|
pub fn LaunchContractSet::profiles(
  self : LaunchContractSet,
) -> ReadOnlyArray[KernelProfile] {
  self.profiles
}

///|
pub fn LaunchContractSet::contracts(
  self : LaunchContractSet,
) -> ReadOnlyArray[AotLaunchContract] {
  self.contracts
}

///|
/// Exact allocation-free lookup over startup-ordered integer identities.
pub fn LaunchContractSet::contract(
  self : LaunchContractSet,
  profile_id : KernelProfileId,
  operation_id : @plan.OperationId,
) -> AotLaunchContract? {
  for contract in self.contracts {
    if contract.profile.id == profile_id &&
      contract.operation_id == operation_id {
      return Some(contract)
    }
  }
  None
}
