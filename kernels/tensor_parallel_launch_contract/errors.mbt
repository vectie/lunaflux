///|
pub(all) enum TensorParallelLaunchContractRule {
  InvalidLimit
  Identity
  Rank
  WorldSize
  DeviceTarget
  CatalogVersion
  OperationCount
  OperationOrder
  OperationShape
  TensorCount
  TensorPlacement
  TensorExtent
  TensorRegion
  FullTensorReplication
  KvLayout
  FullKvReplication
  ProfileCount
  ProfileEnvelope
  DuplicateProfile
  ContractCount
  ContractOrder
  ArtifactProfile
  ArtifactFamily
  VendorImplementation
  Binding
  ExecutionSemantics
  CollectiveSequence
  CollectiveOperation
  CollectiveWidth
  OperandCount
  RankConstantsDestination
  CapacityOverflow
} derive(Debug, Eq)

///|
pub(all) suberror TensorParallelLaunchContractError {
  InvalidTensorParallelLaunchContract(
    rule~ : TensorParallelLaunchContractRule,
    profile~ : Int,
    operation~ : Int
  )
} derive(Debug, Eq)

///|
fn invalid(
  rule : TensorParallelLaunchContractRule,
  profile? : Int = -1,
  operation? : Int = -1,
) -> TensorParallelLaunchContractError {
  InvalidTensorParallelLaunchContract(rule~, profile~, operation~)
}
