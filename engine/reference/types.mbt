///|
/// Caller-owned correctness limits for one offline reference execution.
pub struct ExecutionLimits {
  kernels : @refkernels.ReferenceLimits
  sampling : @sampling.SamplingLimits
  max_operations : Int32
  max_tensors : Int32
  max_input_tokens : Int32
} derive(Debug, Eq)

///|
/// Immutable logits and deterministic greedy selection produced by one plan.
pub struct ExecutionResult {
  identity : @spec.ModelIdentity
  logits : @refkernels.Activation
  selection : @sampling.Selection
  operation_count : Int32
} derive(Debug, Eq)

///|
/// Identity-bound, once-decoded tensors for repeated reference execution.
pub struct PreparedModel {
  identity : @spec.ModelIdentity
  tensors : ReadOnlyArray[@refkernels.Tensor]
} derive(Debug, Eq)

///|
/// Immutable greedy tokens produced by bounded full-prefill recomputation.
pub struct GenerationResult {
  identity : @spec.ModelIdentity
  input_tokens : Int32
  generated_tokens : ReadOnlyArray[Int32]
} derive(Debug, Eq)

///|
/// Constructs an explicit reference-execution envelope.
pub fn ExecutionLimits::new(
  kernels~ : @refkernels.ReferenceLimits,
  sampling~ : @sampling.SamplingLimits,
  max_operations~ : Int32,
  max_tensors~ : Int32,
  max_input_tokens~ : Int32,
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
  { kernels, sampling, max_operations, max_tensors, max_input_tokens, }
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
pub fn ExecutionLimits::max_operations(self : ExecutionLimits) -> Int32 {
  self.max_operations
}

///|
pub fn ExecutionLimits::max_tensors(self : ExecutionLimits) -> Int32 {
  self.max_tensors
}

///|
pub fn ExecutionLimits::max_input_tokens(self : ExecutionLimits) -> Int32 {
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
pub fn ExecutionResult::operation_count(self : ExecutionResult) -> Int32 {
  self.operation_count
}

///|
pub fn PreparedModel::identity(self : PreparedModel) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn PreparedModel::tensor_count(self : PreparedModel) -> Int32 {
  self.tensors.length()
}

///|
pub fn GenerationResult::identity(
  self : GenerationResult,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn GenerationResult::input_tokens(self : GenerationResult) -> Int32 {
  self.input_tokens
}

///|
pub fn GenerationResult::generated_tokens(
  self : GenerationResult,
) -> ReadOnlyArray[Int32] {
  self.generated_tokens
}
