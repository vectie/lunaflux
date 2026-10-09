///|
/// Startup-only semantic requirements for the official MiniMax H3 base
/// pipeline.
///
/// These capabilities describe the complete released FL2VA/Ref2VA system;
/// they are not token-step dispatch tags and never reach scheduler policy.
pub fn requirements() -> @family_contract.ModelFamilyRequirements {
  try! @family_contract.ModelFamilyRequirements::new(
    family=MiniMaxH3,
    architecture="MiniMaxH3DiTModel",
    model_type="minimax_h3",
    schema_version=1,
    workload=AudioVideoDiffusion,
    capabilities=[
      TextConditioningEncoder,
      VisionEncoder,
      MultimodalProjection,
      DiffusionDenoising,
      VideoVae,
      AudioVae,
      ReferenceConditioning,
    ],
  )
}

///|
/// Joins the official H3 requirements with a backend's startup-only semantic
/// capability set. Success grants no execution authority.
pub fn admit_backend(
  backend : @family_contract.ModelBackendCapabilities,
) -> @family_contract.ModelFamilyAdmission {
  requirements().admit(backend)
}
