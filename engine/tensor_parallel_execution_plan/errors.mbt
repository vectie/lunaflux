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
    operation~ : Int,
    operand~ : Int
  )
} derive(Debug, Eq)

///|
fn rejected(
  rule : TensorParallelExecutionPlanRule,
  operation? : Int = -1,
  operand? : Int = -1,
) -> TensorParallelExecutionPlanError {
  InvalidTensorParallelExecutionPlan(rule~, operation~, operand~)
}
