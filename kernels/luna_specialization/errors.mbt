///|
pub(all) enum LunaSpecializationRule {
  Digest
  CompilerIdentity
  CompilerPolicy
  NumericalPolicy
  Evidence
  Identity
  Target
  CatalogVersion
  Operation
  OperationShape
  OperationBinding
  ExecutionProfile
  TensorOffset
  TensorLayout
  Workspace
  Rope
  Quantization
  EntryPoint
  CanonicalBytes
} derive(Debug, Eq)

///|
pub(all) suberror LunaSpecializationError {
  Invalid(LunaSpecializationRule)
  Unsupported(LunaSpecializationRule)
  Mismatch(LunaSpecializationRule)
  SizeOverflow(LunaSpecializationRule)
} derive(Debug, Eq)
