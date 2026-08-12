///|
pub(all) enum ExecutionLimitField {
  Operations
  Tensors
  InputTokens
} derive(Debug, Eq)

///|
pub(all) enum ExecutionGraphFailure {
  IdentityMismatch
  OperationLimitExceeded
  TensorLimitExceeded
  InputLimitExceeded
  InputContextExceeded
  InvalidGenerationLength
  MissingActivation
  MissingTensor
  UnsupportedInput
  ShapeMismatch
  FinalOutputMismatch
} derive(Debug, Eq)

///|
/// Bounded failures from architecture-neutral reference-plan interpretation.
pub(all) suberror ReferenceExecutionError {
  InvalidLimit(ExecutionLimitField)
  InvalidGraph(ExecutionGraphFailure)
  Kernel(@refkernels.ReferenceKernelError)
  Sampling(@sampling.SamplingError)
} derive(Debug, Eq)
