///|
/// RF updates retain F32 state and velocity, with three prepared F32
/// coefficients and separately rounded multiply/add stages (no BF16 cast).
pub(all) enum JointDiffusionCudaNumericalContract {
  OrderedFloat32RfVelocityEtaZeroV1
  OrderedFloat32DenseF32BoundaryBf16RoundV1
  OrderedFloat32DenseF32OutputV1
  OrderedFloat32RmsNormBf16RoundV1
  StagedBf16AdaLnF32BoundaryV1
  SiluF32CastBf16OrderedDenseBf16OutputV1
  BitExactBf16PackedModalityGatherV1
  Float32CosSinExpfPeriod10000V1
  OrderedFloat32DenseSiluDenseV1
  OrderedF32AccumulationBf16SwiGluDownBf16V1
  OrderedF32AccumulationBf16StagedAttentionV1
  OrderedF32ChannelMajorDestandardizationV1
  OrderedF32ChannelMajorPointwiseProjectionV1
  OrderedF32PerOutputWeightNormConv1dV1
  OrderedF32PerInputWeightNormConvTranspose1dV1
  OrderedF32AliasFreeSnakeBetaV1
  OrderedF32AmpParallelResidualAverageV1
} derive(Debug, Eq)

///|
pub(all) enum JointDiffusionCudaAotInvalidKind {
  Limit
  Requirement
  RequestShape
  Target
  Compiler
  Encoding
} derive(Debug, Eq)

///|
pub(all) suberror JointDiffusionCudaAotError {
  Invalid(JointDiffusionCudaAotInvalidKind)
  UnsupportedRequirement(Int)
  SizeOverflow
  Request(@joint_diffusion_plan.JointDiffusionPlanError)
  Operand(
    @joint_diffusion_artifact_admission.JointDiffusionArtifactAdmissionError
  )
  Launch(@launch_contract.LaunchContractError)
  Source(@decoder_projection_cuda_source.DecoderProjectionCudaSourceError)
  FoundationSource(
    @decoder_foundation_cuda_source.DecoderFoundationCudaSourceError
  )
  AffineSource(@affine_modulation_cuda_source.AffineModulationCudaSourceError)
  ChannelMajorSource(
    @channel_major_projection_cuda_source.ChannelMajorProjectionCudaSourceError
  )
  GatedMlpSource(@gated_mlp_cuda_source.GatedMlpCudaSourceError)
  AttentionSource(@joint_attention_cuda_source.JointAttentionCudaSourceError)
  AliasFreeSource(
    @alias_free_activation_cuda_source.AliasFreeActivationCudaSourceError
  )
  JointAudioAmpSource(@joint_audio_amp_cuda_source.JointAudioAmpCudaSourceError)
  AmpParameterLayout(
    @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayoutError
  )
  AmpWorkspaceLayout(
    @joint_audio_amp_workspace_layout.JointAudioAmpWorkspaceLayoutError
  )
} derive(Debug, Eq)

///|
pub struct JointDiffusionCudaAotLimits {
  priv maximum_elements : Int64
  priv maximum_source_bytes : Int
  priv maximum_recipe_bytes : Int
}

///|
pub struct JointDiffusionCudaSourceDigest(String) derive(Debug, Eq, Hash)

///|
pub struct JointDiffusionCudaRecipeDigest(String) derive(Debug, Eq, Hash)

///|
/// Offline-only, request-shape-specialized shifted-flow CUDA candidate. It has
/// no compiled module, loader, launch authority, or qualification claim.
pub struct JointDiffusionShiftedFlowCudaCandidate {
  priv numerical_contract : JointDiffusionCudaNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation
  priv capability : @joint_diffusion_kernel_requirements.JointDiffusionAotCapability
  priv latent : @joint_diffusion_plan.JointDiffusionLatentKind
  priv request_shape : @joint_diffusion_plan.JointDiffusionRequestShape
  priv element_count : Int64
  priv target : @catalog.DeviceTarget
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : JointDiffusionCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : JointDiffusionCudaRecipeDigest
}

///|
/// Offline-only, request-shape-specialized latent projection CUDA candidate.
/// Input rows, row-major `[output,input]` weights, and bias are separate F32
/// buffers; output rows are rounded once to BF16. The
/// candidate has no compiled module, loader, launch authority, or qualification
/// claim.
pub struct JointDiffusionLatentProjectionCudaCandidate {
  priv numerical_contract : JointDiffusionCudaNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation
  priv capability : @joint_diffusion_kernel_requirements.JointDiffusionAotCapability
  priv latent : @joint_diffusion_plan.JointDiffusionLatentKind
  priv request_shape : @joint_diffusion_plan.JointDiffusionRequestShape
  priv row_count : Int64
  priv input_width : Int
  priv output_width : Int
  priv input_element_count : Int64
  priv output_element_count : Int64
  priv projection_weight_element_count : Int64
  priv projection_bias_element_count : Int64
  priv target : @catalog.DeviceTarget
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : JointDiffusionCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : JointDiffusionCudaRecipeDigest
}

///|
/// Offline-only, request-shape-specialized output-head CUDA candidate. The
/// input is the final-normalized F32 hidden rows selected for one modality;
/// weights are row-major F32 `[output,input]`, bias and output are F32. Moving
/// the row selection before this row-independent dense projection preserves
/// the official project-then-select result while avoiding unused conditioning
/// and foreign-modality rows. This grants no compiled module, loader, launch
/// authority, or qualification claim.
pub struct JointDiffusionOutputProjectionCudaCandidate {
  priv numerical_contract : JointDiffusionCudaNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation
  priv capability : @joint_diffusion_kernel_requirements.JointDiffusionAotCapability
  priv latent : @joint_diffusion_plan.JointDiffusionLatentKind
  priv request_shape : @joint_diffusion_plan.JointDiffusionRequestShape
  priv row_count : Int64
  priv input_width : Int
  priv output_width : Int
  priv input_element_count : Int64
  priv output_element_count : Int64
  priv projection_weight_element_count : Int64
  priv projection_bias_element_count : Int64
  priv target : @catalog.DeviceTarget
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : JointDiffusionCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : JointDiffusionCudaRecipeDigest
}

///|
/// Offline-only, request-shape-specialized final RMSNorm candidate for the
/// selected target rows of one modality. Input, `norm_out.norm.weight`, and
/// output are BF16; reduction and scaling use the declared ordered F32
/// contract. AdaLN shift/scale modulation and the later F32 head cast are
/// separate operations and are not claimed here.
pub struct JointDiffusionFinalRmsNormCudaCandidate {
  priv numerical_contract : JointDiffusionCudaNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation
  priv capability : @joint_diffusion_kernel_requirements.JointDiffusionAotCapability
  priv latent : @joint_diffusion_plan.JointDiffusionLatentKind
  priv request_shape : @joint_diffusion_plan.JointDiffusionRequestShape
  priv row_count : Int64
  priv hidden_size : Int
  priv epsilon_numerator : Int
  priv epsilon_denominator : Int
  priv element_count : Int64
  priv weight_element_count : Int64
  priv target : @catalog.DeviceTarget
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : JointDiffusionCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : JointDiffusionCudaRecipeDigest
}

///|
/// Consumes already row-aligned BF16 normalized, shift, and scale buffers,
/// performs staged BF16 AdaLN affine arithmetic, and widens the final BF16
/// result to F32 for one output head. Producing shift/scale from `temb` and
/// selecting them through packed row-to-timestep indices remain unimplemented.
pub struct JointDiffusionFinalAdaLnF32CudaCandidate {
  priv numerical_contract : JointDiffusionCudaNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation
  priv capability : @joint_diffusion_kernel_requirements.JointDiffusionAotCapability
  priv latent : @joint_diffusion_plan.JointDiffusionLatentKind
  priv request_shape : @joint_diffusion_plan.JointDiffusionRequestShape
  priv row_count : Int64
  priv hidden_size : Int
  priv element_count : Int64
  priv target : @catalog.DeviceTarget
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : JointDiffusionCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : JointDiffusionCudaRecipeDigest
}

///|
pub struct JointDiffusionFinalAdaLnParameterRowCudaCandidate {
  priv numerical_contract : JointDiffusionCudaNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation
  priv capability : @joint_diffusion_kernel_requirements.JointDiffusionAotCapability
  priv input_width : Int
  priv hidden_size : Int
  priv weight_element_count : Int64
  priv target : @catalog.DeviceTarget
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : JointDiffusionCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : JointDiffusionCudaRecipeDigest
}

///|
pub struct JointDiffusionPackedModalityAdaLnCudaCandidate {
  priv numerical_contract : JointDiffusionCudaNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation
  priv capability : @joint_diffusion_kernel_requirements.JointDiffusionAotCapability
  priv timestep_digest : @spec.PlanDigest
  priv modality_digest : @spec.PlanDigest
  priv hidden_size : Int
  priv packed_rows : Int
  priv video_rows : Int
  priv audio_rows : Int
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
  ]
  priv source : Bytes
  priv source_digest : JointDiffusionCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : JointDiffusionCudaRecipeDigest
}

///|
pub struct JointDiffusionTimestepFrequencyCudaCandidate {
  priv numerical_contract : JointDiffusionCudaNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation
  priv capability : @joint_diffusion_kernel_requirements.JointDiffusionAotCapability
  priv values_digest : @spec.PlanDigest
  priv row_count : Int
  priv frequency_width : Int
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
  ]
  priv source : Bytes
  priv source_digest : JointDiffusionCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : JointDiffusionCudaRecipeDigest
}

///|
pub struct JointDiffusionTimestepMlpCudaCandidate {
  priv numerical_contract : JointDiffusionCudaNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation
  priv capability : @joint_diffusion_kernel_requirements.JointDiffusionAotCapability
  priv values_digest : @spec.PlanDigest
  priv row_count : Int
  priv input_width : Int
  priv hidden_width : Int
  priv output_width : Int
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
  ]
  priv source : Bytes
  priv source_digest : JointDiffusionCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : JointDiffusionCudaRecipeDigest
}

///|
/// Request-bound parameter-table production for every authenticated distinct
/// timestep row. It remains offline-only and grants no launch authority.
pub struct JointDiffusionDistinctTimestepAdaLnCudaCandidate {
  priv numerical_contract : JointDiffusionCudaNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation
  priv capability : @joint_diffusion_kernel_requirements.JointDiffusionAotCapability
  priv values_digest : @spec.PlanDigest
  priv row_count : Int
  priv input_width : Int
  priv hidden_size : Int
  priv weight_element_count : Int64
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
  ]
  priv source : Bytes
  priv source_digest : JointDiffusionCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : JointDiffusionCudaRecipeDigest
}

///|
/// An inert validated join of the three distinct-timestep conditioning stages.
pub struct JointDiffusionTimestepConditioningComposition {
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest
  priv values_digest : @spec.PlanDigest
  priv row_count : Int
  priv frequency_width : Int
  priv mlp_hidden_width : Int
  priv conditioning_width : Int
  priv adaln_hidden_size : Int
}

///|
/// Offline composite source for all 50 packed-row and 2 text-refiner SwiGLU
/// layers. Both row scopes share exact weight geometry but separate launches.
pub struct JointDiffusionTransformerMlpCudaCandidate {
  priv numerical_contract : JointDiffusionCudaNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest
  priv request_digest : @spec.PlanDigest
  priv requirement_ordinal : Int
  priv operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation
  priv capability : @joint_diffusion_kernel_requirements.JointDiffusionAotCapability
  priv layer_count : Int
  priv refiner_layer_count : Int
  priv packed_row_count : Int
  priv refiner_row_count : Int
  priv hidden_size : Int
  priv intermediate_size : Int
  priv packed_weight_element_count : Int64
  priv down_weight_element_count : Int64
  priv packed_dimensions : @launch_contract.AotLaunchDimensions
  priv refiner_dimensions : @launch_contract.AotLaunchDimensions
  priv packed_operands : ReadOnlyArray[
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
  ]
  priv refiner_operands : ReadOnlyArray[
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
  ]
  priv source : Bytes
  priv source_digest : JointDiffusionCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : JointDiffusionCudaRecipeDigest
}

///|
pub(all) enum JointDiffusionAttentionRowScope {
  PackedRows
  RefinerRows
} derive(Debug, Eq)

///|
pub(all) enum JointDiffusionAttentionStageKind {
  QkvProjection
  QkNormRotary
  QkNorm
  FullAttention
  OutputProjection
} derive(Debug, Eq)

///|
pub struct JointDiffusionAttentionCudaStage {
  priv scope : JointDiffusionAttentionRowScope
  priv kind : JointDiffusionAttentionStageKind
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
  ]
}

///|
/// Eight explicit staged entrypoints cover packed rotary attention for the 50
/// main blocks and non-rotary text attention for the 2 refiner blocks.
pub struct JointDiffusionTransformerAttentionCudaCandidate {
  priv numerical_contract : JointDiffusionCudaNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest
  priv request_digest : @spec.PlanDigest
  priv requirement_ordinal : Int
  priv operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation
  priv capability : @joint_diffusion_kernel_requirements.JointDiffusionAotCapability
  priv layer_count : Int
  priv refiner_layer_count : Int
  priv packed_row_count : Int
  priv refiner_row_count : Int
  priv hidden_size : Int
  priv head_count : Int
  priv head_dimension : Int
  priv inner_size : Int
  priv rotary_dimension : Int
  priv qk_epsilon_numerator : Int
  priv qk_epsilon_denominator : Int
  priv projection_weight_element_count : Int64
  priv norm_weight_element_count : Int64
  priv stages : ReadOnlyArray[JointDiffusionAttentionCudaStage]
  priv source : Bytes
  priv source_digest : JointDiffusionCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : JointDiffusionCudaRecipeDigest
}

///|
/// Offline-only request-bound inverse video-latent normalization. This exact
/// F32 prefix ends before the decoder dtype cast and post-quant convolution.
pub struct JointDiffusionVideoVaeLatentDestandardizationCudaCandidate {
  priv numerical_contract : JointDiffusionCudaNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation
  priv capability : @joint_diffusion_kernel_requirements.JointDiffusionAotCapability
  priv parameter_digest : @joint_diffusion_vae_reference.JointVaeLatentDestandardizationDigest
  priv request_shape : @joint_diffusion_plan.JointDiffusionRequestShape
  priv batch_count : Int
  priv channel_count : Int
  priv elements_per_channel : Int64
  priv element_count : Int64
  priv target : @catalog.DeviceTarget
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : JointDiffusionCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : JointDiffusionCudaRecipeDigest
}
