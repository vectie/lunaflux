///|
/// Diffusion semantics intentionally have no alias in the current generic
/// decoder kernel vocabulary. This method always fails closed.
pub fn JointDiffusionKernelRequirement::require_current_generic_decoder_capability(
  self : Self,
  ordinal~ : Int,
) -> @plan.KernelCapabilityId raise JointDiffusionKernelRequirementsError {
  raise UnsupportedGenericDecoderKernelOperation(ordinal, self.operation)
}

///|
/// Rejects attempts to weaken the ordered diffusion requirement set into the
/// current decoder-only generic capability vocabulary.
pub fn JointDiffusionKernelRequirements::require_current_generic_decoder_capabilities(
  self : Self,
) -> ReadOnlyArray[@plan.KernelCapabilityId] raise JointDiffusionKernelRequirementsError {
  let capabilities : Array[@plan.KernelCapabilityId] = []
  for ordinal, requirement in self.requirements {
    capabilities.push(
      requirement.require_current_generic_decoder_capability(ordinal~),
    )
  }
  ReadOnlyArray::from_array(capabilities)
}
