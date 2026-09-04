///|
pub(all) enum AttentionIterationKind {
  AttentionParallelMap
  AttentionSequentialFold
  AttentionAssociativeFold
  AttentionVectorMap
} derive(Debug, Eq)

///|
pub(all) enum AttentionIterationAxis {
  AttentionQueryRowAxis
  AttentionQueryHeadAxis
  AttentionKeyValuePartitionAxis
  AttentionKeyValueAxis
  AttentionSoftmaxPartitionAxis
  AttentionHeadColumnAxis
} derive(Debug, Eq)

///|
/// A functional loop combinator. It describes iteration semantics without
/// assigning CUDA blocks, HIP workgroups, CPU threads, or vendor instructions.
pub struct AttentionLoopCombinator {
  priv kind : AttentionIterationKind
  priv axis : AttentionIterationAxis
  priv extent : Int
  priv tile : Int
} derive(Debug, Eq)

///|
pub struct AttentionTileScheduleDigest(String) derive(Debug, Eq, Hash)

///|
/// Immutable schedule IR derived from a semantic tile program. Backends may
/// refine this schedule but cannot change its pure map/fold meaning.
pub struct AttentionTileSchedule {
  priv program_digest : @luna_attention_tile_ir.AttentionTileProgramDigest
  priv loops : ReadOnlyArray[AttentionLoopCombinator]
  priv compute : @luna_attention_strategy.AttentionTileCompute
  priv memory : @luna_attention_strategy.AttentionTileMemory
  priv subgroups_per_block : Int
  priv pipeline_stages : Int
  priv vector_bytes : Int
  priv shared_memory_bytes : Int64
  priv key_value_partitions : Int
  priv canonical : Bytes
  priv digest : AttentionTileScheduleDigest
} derive(Debug, Eq)

///|
pub(all) enum AttentionTileScheduleRule {
  InvalidProblem
  InvalidTile
  InvalidVector
  InvalidLoopNest
} derive(Debug, Eq)

///|
pub(all) suberror AttentionTileScheduleError {
  InvalidAttentionTileSchedule(AttentionTileScheduleRule)
} derive(Debug, Eq)
