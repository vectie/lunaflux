///|
pub(all) enum QkNormRopePagedKvWriteAotRule {
  Limits
  SemanticPlan
  OperationSpan
  Shape
  Layout
  Profile
  Target
  Compiler
  OperandOrder
  OperandSize
  OperandAlignment
  AliasContract
  Source
  Recipe
  CompiledModule
  CompileReceipt
  Binding
  Determinism
  Reference
} derive(Debug, Eq)

///|
pub(all) suberror QkNormRopePagedKvWriteAotError {
  Invalid(QkNormRopePagedKvWriteAotRule)
  Mismatch(QkNormRopePagedKvWriteAotRule)
  Unsupported(QkNormRopePagedKvWriteAotRule)
  LimitExceeded(QkNormRopePagedKvWriteAotRule)
  SizeOverflow(QkNormRopePagedKvWriteAotRule)
} derive(Debug, Eq)
