///|
/// The only admitted sampling placement for the current semantic graph.
///
/// This is execution ownership, not a CUDA kernel family or manifest
/// capability. Selection happens after bounded terminal-logit readback.
pub(all) enum SamplingExecutionOwner {
  HostCompletionOwner
} derive(Debug, Eq)

///|
/// Typed proof that one model plan terminates in BF16 vocabulary logits and
/// therefore requires no sampling-reduction CUDA artifact.
pub struct HostSamplingBoundary {
  priv owner : SamplingExecutionOwner
  priv producer : @plan.OperationId
  priv final_output : @plan.ActivationRef
  priv vocabulary_size : Int
  priv logits_row_bytes : Int64
} derive(Debug, Eq)

///|
pub fn HostSamplingBoundary::owner(
  self : HostSamplingBoundary,
) -> SamplingExecutionOwner {
  self.owner
}

///|
pub fn HostSamplingBoundary::producer(
  self : HostSamplingBoundary,
) -> @plan.OperationId {
  self.producer
}

///|
pub fn HostSamplingBoundary::final_output(
  self : HostSamplingBoundary,
) -> @plan.ActivationRef {
  self.final_output
}

///|
pub fn HostSamplingBoundary::vocabulary_size(
  self : HostSamplingBoundary,
) -> Int {
  self.vocabulary_size
}

///|
pub fn HostSamplingBoundary::logits_row_bytes(
  self : HostSamplingBoundary,
) -> Int64 {
  self.logits_row_bytes
}

///|
/// The audited current graph never requires a sampling CUDA artifact.
pub fn HostSamplingBoundary::requires_cuda_artifact(
  _self : HostSamplingBoundary,
) -> Bool {
  false
}
