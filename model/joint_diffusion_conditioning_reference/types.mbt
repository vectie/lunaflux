///|
/// Exact missing prerequisites which prevent an ordered conditioning
/// requirement from becoming an AOT kernel candidate. These are semantic
/// gaps, not qualification failures of an otherwise executable artifact.
pub(all) enum JointConditioningAotGap {
  DecodedMediaPayloadBuffers
  VideoVaeEncodeExecution
  VideoVaePosteriorSampling
  VideoVaeLatentNormalization
  AudioVaeEncodeExecution
  AudioVaeLatentNormalization
  ConditioningNoiseValues
  ConditioningScaleNoiseArithmetic
  PackedConditioningRowSpans
  PerReferenceEncodedLatentGeometry
  ReferenceVideoSoundtrackAssociation
} derive(Debug, Eq)

///|
pub(all) suberror JointConditioningReferenceError {
  InvalidLimit
  InvalidRequirements
  InvalidRequest
  UnsupportedRequirement(Int)
  GeometryOverflow
  GeometryBudgetExceeded
  ConditioningSemanticUnavailable(
    @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation,
    JointConditioningAotGap
  )
} derive(Debug, Eq)

///|
pub struct JointConditioningReferenceLimits {
  priv maximum_conditioning_rows : Int64
  priv maximum_conditioning_elements : Int64
}

///|
/// Request-bound geometry which is already derivable without encoding media.
/// Ref2VA row geometry intentionally remains absent because the official row
/// count comes from each encoded reference tensor.
pub struct JointConditioningGeometry {
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest
  priv request_digest : @spec.PlanDigest
  priv requirement_ordinal : Int
  priv operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation
  priv workflow : @joint_diffusion_plan.JointDiffusionWorkflow
  priv keyframe_count : Int
  priv image_reference_count : Int
  priv video_reference_count : Int
  priv audio_reference_count : Int
  priv visual_latent_channels : Int
  priv visual_row_width : Int?
  priv visual_rows_per_condition : Int64?
  priv total_visual_conditioning_rows : Int64?
  priv total_visual_conditioning_elements : Int64?
}

///|
pub struct JointConditioningAotGapDigest {
  priv hex : String
} derive(Debug, Eq, Hash)

///|
/// A semantic assessment only. It contains no CUDA source, compiled module,
/// entry point, launch contract, device target, compiler policy, or artifact
/// admission authority.
pub struct JointConditioningAotAssessment {
  priv geometry : JointConditioningGeometry
  priv gap_digest : JointConditioningAotGapDigest
  priv gaps : ReadOnlyArray[JointConditioningAotGap]
}
