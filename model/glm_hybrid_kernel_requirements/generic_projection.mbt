///|
pub fn GlmHybridKernelRequirement::current_generic_capability(
  self : Self,
  ordinal~ : Int,
) -> @plan.KernelCapabilityId raise GlmHybridKernelRequirementsError {
  match self.operation {
    TokenEmbedding => @plan.KernelCapabilityId::embedding_lookup()
    DecoderRmsNorm(_, _) | FinalRmsNorm(_, _) =>
      @plan.KernelCapabilityId::rms_norm()
    QueryKeyRmsNorm(_) => @plan.KernelCapabilityId::qk_rms_norm()
    LanguageModelHead(_, _) => @plan.KernelCapabilityId::dense_projection()
    operation => raise UnsupportedGenericKernelOperation(ordinal, operation)
  }
}

///|
pub fn GlmHybridKernelRequirements::require_current_generic_capabilities(
  self : Self,
) -> ReadOnlyArray[@plan.KernelCapabilityId] raise GlmHybridKernelRequirementsError {
  let projected : Array[@plan.KernelCapabilityId] = []
  for ordinal, requirement in self.requirements {
    let capability = requirement.current_generic_capability(ordinal~)
    if !projected.contains(capability) {
      projected.push(capability)
    }
  }
  ReadOnlyArray::from_array(projected)
}
