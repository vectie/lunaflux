///|
pub(all) enum PagedAttentionAotRule {
  Limits
  Contract
  Operation
  Capability
  SemanticVersion
  ExecutionSemantics
  Shape
  Profile
  Dimensions
  OperandOrder
  OperandBytes
  KvLayout
  DeviceTarget
  CompilerPolicy
  Source
  Recipe
} derive(Debug, Eq)

///|
pub(all) suberror PagedAttentionAotError {
  Invalid(PagedAttentionAotRule)
  Unsupported(PagedAttentionAotRule)
  Mismatch(PagedAttentionAotRule)
  LimitExceeded(PagedAttentionAotRule)
  SizeOverflow(PagedAttentionAotRule)
} derive(Debug, Eq)
