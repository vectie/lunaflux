///|
pub(all) enum PagedKvWriteAotRule {
  Limits
  Identity
  Operation
  Shape
  Layout
  Profile
  Target
  Compiler
  OperandOrder
  OperandSize
  OperandAlignment
  ActivationAlias
  Fallback
  Source
  Recipe
  KernelAbi
  CompiledArtifact
  CompileReceipt
  Determinism
  Digest
} derive(Debug, Eq)

///|
pub(all) suberror PagedKvWriteAotError {
  Invalid(PagedKvWriteAotRule)
  Mismatch(PagedKvWriteAotRule)
  Unsupported(PagedKvWriteAotRule)
  LimitExceeded(PagedKvWriteAotRule)
  SizeOverflow(PagedKvWriteAotRule)
} derive(Debug, Eq)
