///|
/// Backend-neutral semantic fusion families. Device backends may lower these
/// plans, but model builders and schedulers never name a vendor implementation.
pub(all) enum FusionSemanticFamily {
  QkvQkNormPositionedRopePagedKvWrite
} derive(Debug, Eq)

///|
pub(all) enum FusionPlanRule {
  OperationOrder
  Dataflow
  Shape
  PersistentState
} derive(Debug, Eq)

///|
pub(all) suberror FusionPlanError {
  InvalidFusionPlan(FusionPlanRule)
} derive(Debug, Eq)

///|
/// Exact Qwen-style attention ingress fusion. This is semantic IR only: it
/// contains no CUDA names, warp width, block size, or device instruction.
pub struct QkvQkNormPositionedRopePagedKvWritePlan {
  priv qkv_operation : @plan.OperationId
  priv qk_norm_operation : @plan.OperationId
  priv rope_operation : @plan.OperationId
  priv attention_operation : @plan.OperationId
  priv kv_layer : @plan.KvLayerId
  priv input_width : Int
  priv output_width : Int
  priv query_heads : Int
  priv kv_heads : Int
  priv head_dimension : Int
  priv norm_epsilon : Double
  priv rope_theta : Double
  priv projected_qkv : @plan.ActivationRef
  priv positioned_qkv : @plan.ActivationRef
}
