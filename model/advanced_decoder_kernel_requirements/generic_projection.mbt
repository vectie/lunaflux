///|
/// Projects one requirement only where the current dense generic vocabulary
/// expresses identical behavior. Similar-looking advanced operations are not
/// weakened to dense kernels.
pub fn AdvancedDecoderKernelRequirement::current_generic_capability(
  self : Self,
  ordinal~ : Int,
) -> @plan.KernelCapabilityId raise AdvancedDecoderKernelRequirementsError {
  match self.operation {
    TokenEmbedding => @plan.KernelCapabilityId::embedding_lookup()
    FinalRmsNorm(_) => @plan.KernelCapabilityId::rms_norm()
    LanguageModelHead => @plan.KernelCapabilityId::dense_projection()
    operation => raise UnsupportedGenericKernelOperation(ordinal, operation)
  }
}

///|
/// Requires a lossless projection of the entire ordered set into the current
/// dense generic kernel vocabulary. Every advanced decoder plan currently
/// fails closed; this function does not claim runnable artifacts.
pub fn AdvancedDecoderKernelRequirements::require_current_generic_capabilities(
  self : Self,
) -> ReadOnlyArray[@plan.KernelCapabilityId] raise AdvancedDecoderKernelRequirementsError {
  let projected : Array[@plan.KernelCapabilityId] = []
  for ordinal, requirement in self.requirements {
    let capability = requirement.current_generic_capability(ordinal~)
    if !projected.contains(capability) {
      projected.push(capability)
    }
  }
  ReadOnlyArray::from_array(projected)
}
