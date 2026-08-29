///|
pub(all) enum PagedAttentionParallelAotRule {
  Limits
  ReferenceCandidate
  Shape
  Profile
  KvLayout
  DeviceTarget
  CompilerPolicy
  LunaTile
  StaticResources
  Source
  Recipe
  Digest
  PromotionEvidence
  ObservationOrder
  Correctness
  MicrobenchmarkWin
  EndToEndRegression
  CompiledEntryPoint
} derive(Debug, Eq)

///|
pub(all) suberror PagedAttentionParallelAotError {
  PagedAttentionParallelInvalid(PagedAttentionParallelAotRule)
  PagedAttentionParallelUnsupported(PagedAttentionParallelAotRule)
  PagedAttentionParallelMismatch(PagedAttentionParallelAotRule)
  PagedAttentionParallelLimitExceeded(PagedAttentionParallelAotRule)
  PagedAttentionParallelOverflow(PagedAttentionParallelAotRule)
} derive(Debug, Eq)
