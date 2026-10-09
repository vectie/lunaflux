///|
/// Returns the exact versioned profiles registered by the modular family
/// packages. Ordering is stable evidence; callers must not infer priority from
/// it.
pub fn profiles() -> ReadOnlyArray[@family_contract.ModelFamilyRequirements] {
  [
    @deepseek_v4.requirements(),
    @minimax_h3.requirements(),
    @glm53.full_requirements(),
    @glm53.flash_requirements(),
  ]
}

///|
/// Semantic capability set of the established dense decoder graph.
///
/// This intentionally excludes in-progress or qualification-only operations.
/// A family becomes executable only when its complete production backend
/// supplies a separately constructed capability set.
pub fn baseline_dense_decoder_backend() -> @family_contract.ModelBackendCapabilities {
  try! @family_contract.ModelBackendCapabilities::new([DenseCausalAttention])
}

///|
/// Joins every registered profile with one startup backend. Family branching
/// ends here; scheduler and KV packages never import this catalog.
pub fn admit_profiles(
  backend : @family_contract.ModelBackendCapabilities,
) -> ReadOnlyArray[@family_contract.ModelFamilyAdmission] {
  let admissions : Array[@family_contract.ModelFamilyAdmission] = []
  for profile in profiles() {
    admissions.push(profile.admit(backend))
  }
  ReadOnlyArray::from_array(admissions)
}
