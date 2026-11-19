///|
pub(all) enum TensorParallelExecutionPlanRule {
  InvalidLimit
  ModelIdentity
  Rank
  WorldSize
  DeviceTarget
  CatalogVersion
  Profile
  OperationCount
  OperationOrder
  OperationKind
  OperationCapability
  ArtifactBundle
  ArtifactEntry
  ArtifactModule
  Operand
  Descriptor
  RankConstants
  WeightRegion
  FullShardedWeight
  KvRegion
  ActivationRegion
  WorkspaceRegion
  CollectiveCount
  CollectiveSequence
  CollectiveGeometry
  Alignment
  CapacityOverflow
} derive(Debug, Eq)

///|
pub(all) suberror TensorParallelExecutionPlanError {
  InvalidTensorParallelExecutionPlan(
    rule~ : TensorParallelExecutionPlanRule,
    operation~ : Int32,
    operand~ : Int32
  )
} derive(Debug, Eq)

///|
fn rejected(
  rule : TensorParallelExecutionPlanRule,
  operation~ : Int32 = -1,
  operand~ : Int32 = -1,
) -> TensorParallelExecutionPlanError {
  InvalidTensorParallelExecutionPlan(rule~, operation~, operand~)
}
