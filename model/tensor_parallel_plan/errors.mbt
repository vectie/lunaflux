///|
pub(all) enum TensorParallelPlanRule {
  Topology
  Alignment
  Extent
  Region
  TransferRecipe
  TransferSegment
  TensorOrder
  OperationOrder
  OperationGeometry
  UnsupportedSemanticOperation
  CollectiveOrder
  CollectiveGeometry
  ArenaBytes
  MaterializedBytes
  ArithmeticOverflow
} derive(Debug, Eq)

///|
pub(all) suberror TensorParallelPlanError {
  InvalidTensorParallelPlan(rule~ : TensorParallelPlanRule, index~ : Int)
} derive(Debug, Eq)

///|
fn invalid(
  rule : TensorParallelPlanRule,
  index? : Int = -1,
) -> TensorParallelPlanError {
  InvalidTensorParallelPlan(rule~, index~)
}
