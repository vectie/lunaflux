///|
/// Caller-owned correctness limits for one offline reference execution.
pub struct ExecutionLimits {
  kernels : @refkernels.ReferenceLimits
  sampling : @sampling.SamplingLimits
  max_operations : Int
  max_tensors : Int
  max_input_tokens : Int
} derive(Debug, Eq)

///|
/// Immutable logits and deterministic greedy selection produced by one plan.
pub struct ExecutionResult {
  identity : @spec.ModelIdentity
  logits : @refkernels.Activation
  selection : @sampling.Selection
  operation_count : Int
} derive(Debug, Eq)

///|
/// Identity-bound, once-decoded tensors for repeated reference execution.
pub struct PreparedModel {
  identity : @spec.ModelIdentity
  tensors : ReadOnlyArray[@refkernels.Tensor]
} derive(Debug, Eq)

///|
/// Constructs an explicit reference-execution envelope.
pub fn ExecutionLimits::new(
  kernels~ : @refkernels.ReferenceLimits,
  sampling~ : @sampling.SamplingLimits,
  max_operations~ : Int,
  max_tensors~ : Int,
  max_input_tokens~ : Int,
) -> ExecutionLimits raise ReferenceExecutionError {
  if max_operations <= 0 {
    raise InvalidLimit(Operations)
  }
  if max_tensors <= 0 {
    raise InvalidLimit(Tensors)
  }
  if max_input_tokens <= 0 {
    raise InvalidLimit(InputTokens)
  }
  { kernels, sampling, max_operations, max_tensors, max_input_tokens }
}

///|
pub fn ExecutionLimits::kernels(
  self : ExecutionLimits,
) -> @refkernels.ReferenceLimits {
  self.kernels
}

///|
pub fn ExecutionLimits::sampling(
  self : ExecutionLimits,
) -> @sampling.SamplingLimits {
  self.sampling
}

///|
pub fn ExecutionLimits::max_operations(self : ExecutionLimits) -> Int {
  self.max_operations
}

///|
pub fn ExecutionLimits::max_tensors(self : ExecutionLimits) -> Int {
  self.max_tensors
}

///|
pub fn ExecutionLimits::max_input_tokens(self : ExecutionLimits) -> Int {
  self.max_input_tokens
}

///|
pub fn ExecutionResult::identity(self : ExecutionResult) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn ExecutionResult::logits(
  self : ExecutionResult,
) -> @refkernels.Activation {
  self.logits
}

///|
pub fn ExecutionResult::selection(
  self : ExecutionResult,
) -> @sampling.Selection {
  self.selection
}

///|
pub fn ExecutionResult::operation_count(self : ExecutionResult) -> Int {
  self.operation_count
}

///|
pub fn PreparedModel::identity(self : PreparedModel) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn PreparedModel::tensor_count(self : PreparedModel) -> Int {
  self.tensors.length()
}
