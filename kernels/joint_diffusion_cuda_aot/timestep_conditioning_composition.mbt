///|
pub fn compose_timestep_conditioning_candidates(
  frequency : JointDiffusionTimestepFrequencyCudaCandidate,
  mlp : JointDiffusionTimestepMlpCudaCandidate,
  adaln : JointDiffusionDistinctTimestepAdaLnCudaCandidate,
) -> JointDiffusionTimestepConditioningComposition raise JointDiffusionCudaAotError {
  if frequency.source_identity() != mlp.source_identity() ||
    frequency.source_identity() != adaln.source_identity() ||
    frequency.requirements_digest() != mlp.requirements_digest() ||
    frequency.requirements_digest() != adaln.requirements_digest() ||
    frequency.values_digest() != mlp.values_digest() ||
    frequency.values_digest() != adaln.values_digest() {
    raise Invalid(RequestShape)
  }
  if frequency.row_count() != mlp.row_count() ||
    frequency.row_count() != adaln.row_count() {
    raise Invalid(RequestShape)
  }
  if frequency.requirement_ordinal() >= mlp.requirement_ordinal() ||
    mlp.requirement_ordinal() >= adaln.requirement_ordinal() ||
    frequency.frequency_width() != mlp.input_width() ||
    mlp.output_width() != adaln.input_width() {
    raise Invalid(Requirement)
  }
  {
    source_identity: frequency.source_identity(),
    requirements_digest: frequency.requirements_digest(),
    values_digest: frequency.values_digest(),
    row_count: frequency.row_count(),
    frequency_width: frequency.frequency_width(),
    mlp_hidden_width: mlp.hidden_width(),
    conditioning_width: mlp.output_width(),
    adaln_hidden_size: adaln.hidden_size(),
  }
}

///|
pub fn JointDiffusionTimestepConditioningComposition::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn JointDiffusionTimestepConditioningComposition::requirements_digest(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn JointDiffusionTimestepConditioningComposition::values_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.values_digest
}

///|
pub fn JointDiffusionTimestepConditioningComposition::row_count(
  self : Self,
) -> Int {
  self.row_count
}

///|
pub fn JointDiffusionTimestepConditioningComposition::frequency_width(
  self : Self,
) -> Int {
  self.frequency_width
}

///|
pub fn JointDiffusionTimestepConditioningComposition::mlp_hidden_width(
  self : Self,
) -> Int {
  self.mlp_hidden_width
}

///|
pub fn JointDiffusionTimestepConditioningComposition::conditioning_width(
  self : Self,
) -> Int {
  self.conditioning_width
}

///|
pub fn JointDiffusionTimestepConditioningComposition::adaln_hidden_size(
  self : Self,
) -> Int {
  self.adaln_hidden_size
}

///|
pub fn JointDiffusionTimestepConditioningComposition::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}
