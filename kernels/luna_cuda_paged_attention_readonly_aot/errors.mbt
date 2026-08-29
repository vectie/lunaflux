///|
pub(all) enum PagedAttentionReadOnlyRule {
  Limits
  Reference
  Operation
  Profile
  KvLayout
  DeviceTarget
  CompilerPolicy
  KernelAbi
  Symbol
  Source
  Recipe
  Digest
  CompiledArtifact
  CompileReceipt
  Determinism
  Evidence
} derive(Debug, Eq)

///|
pub(all) suberror PagedAttentionReadOnlyAotError {
  PagedAttentionReadOnlyInvalid(PagedAttentionReadOnlyRule)
  PagedAttentionReadOnlyMismatch(PagedAttentionReadOnlyRule)
  PagedAttentionReadOnlyLimitExceeded(PagedAttentionReadOnlyRule)
}
