///|
/// Startup semantic requirements for the official full GLM-5.3 profile.
///
/// This record is not a plan and grants no execution authority. In
/// particular, the existing dense decoder backend lacks MoE and DSA and must
/// return `MissingCapabilities` instead of silently reusing a Llama graph.
pub fn full_requirements() -> @family_contract.ModelFamilyRequirements {
  try! @family_contract.ModelFamilyRequirements::new(
    family=Glm53,
    architecture="GlmMoeDsaForCausalLM",
    model_type="glm_moe_dsa",
    schema_version=1,
    workload=DecoderOnlyText,
    capabilities=[
      MixtureOfExperts,
      MultiTokenPrediction,
      DynamicSparseAttention,
    ],
  )
}

///|
/// Startup semantic requirements for the official GLM-5.3-Flash profile.
pub fn flash_requirements() -> @family_contract.ModelFamilyRequirements {
  try! @family_contract.ModelFamilyRequirements::new(
    family=Glm53,
    architecture="Glm5NextForConditionalGeneration",
    model_type="glm5_next",
    schema_version=1,
    workload=MultimodalConditionalGeneration,
    capabilities=[
      MixtureOfExperts,
      MultiTokenPrediction,
      HyperConnections,
      DynamicSparseAttention,
      LinearRecurrentAttention,
      VisionEncoder,
      MultimodalProjection,
    ],
  )
}

///|
/// Checks semantic capability names only; it grants no execution authority.
pub fn admit_full_backend(
  metadata : @glm53_spec.Glm53FullMetadata,
  backend : @family_contract.ModelBackendCapabilities,
) -> @family_contract.ModelFamilyAdmission {
  ignore(metadata)
  full_requirements().admit(backend)
}

///|
/// Checks semantic capability names only; it grants no execution authority.
pub fn admit_flash_backend(
  metadata : @glm53_spec.Glm53FlashMetadata,
  backend : @family_contract.ModelBackendCapabilities,
) -> @family_contract.ModelFamilyAdmission {
  ignore(metadata)
  flash_requirements().admit(backend)
}
