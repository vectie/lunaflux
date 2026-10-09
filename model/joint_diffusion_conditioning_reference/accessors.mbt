///|
pub fn JointConditioningReferenceLimits::new(
  maximum_conditioning_rows~ : Int64,
  maximum_conditioning_elements~ : Int64,
) -> Self raise JointConditioningReferenceError {
  if maximum_conditioning_rows <= 0L || maximum_conditioning_elements <= 0L {
    raise InvalidLimit
  }
  { maximum_conditioning_rows, maximum_conditioning_elements, }
}

///|
pub fn JointConditioningAotAssessment::geometry(
  self : Self,
) -> JointConditioningGeometry {
  self.geometry
}

///|
pub fn JointConditioningAotAssessment::gap_digest(
  self : Self,
) -> JointConditioningAotGapDigest {
  self.gap_digest
}

///|
pub fn JointConditioningAotAssessment::gap_count(self : Self) -> Int {
  self.gaps.length()
}

///|
pub fn JointConditioningAotAssessment::gap(
  self : Self,
  index : Int,
) -> JointConditioningAotGap? {
  self.gaps.get(index)
}

///|
/// Always false for this reference-only assessment.
pub fn JointConditioningAotAssessment::aot_candidate_available(
  _self : Self,
) -> Bool {
  false
}

///|
pub fn JointConditioningAotGapDigest::as_hex(self : Self) -> String {
  self.hex
}

///|
pub fn JointConditioningGeometry::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn JointConditioningGeometry::requirements_digest(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn JointConditioningGeometry::request_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.request_digest
}

///|
pub fn JointConditioningGeometry::requirement_ordinal(self : Self) -> Int {
  self.requirement_ordinal
}

///|
pub fn JointConditioningGeometry::operation(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation {
  self.operation
}

///|
pub fn JointConditioningGeometry::workflow(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionWorkflow {
  self.workflow
}

///|
pub fn JointConditioningGeometry::keyframe_count(self : Self) -> Int {
  self.keyframe_count
}

///|
pub fn JointConditioningGeometry::image_reference_count(self : Self) -> Int {
  self.image_reference_count
}

///|
pub fn JointConditioningGeometry::video_reference_count(self : Self) -> Int {
  self.video_reference_count
}

///|
pub fn JointConditioningGeometry::audio_reference_count(self : Self) -> Int {
  self.audio_reference_count
}

///|
pub fn JointConditioningGeometry::visual_latent_channels(self : Self) -> Int {
  self.visual_latent_channels
}

///|
pub fn JointConditioningGeometry::visual_row_width(self : Self) -> Int? {
  self.visual_row_width
}

///|
pub fn JointConditioningGeometry::visual_rows_per_condition(
  self : Self,
) -> Int64? {
  self.visual_rows_per_condition
}

///|
pub fn JointConditioningGeometry::total_visual_conditioning_rows(
  self : Self,
) -> Int64? {
  self.total_visual_conditioning_rows
}

///|
pub fn JointConditioningGeometry::total_visual_conditioning_elements(
  self : Self,
) -> Int64? {
  self.total_visual_conditioning_elements
}
