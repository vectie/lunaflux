///|
pub fn lower_audio_vae_decoder_amp_stage_cuda_aot_candidate(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  request~ : @joint_diffusion_plan.JointDiffusionRequestShape,
  parameter_layout~ : @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayout,
  workspace_layout~ : @joint_audio_amp_workspace_layout.JointAudioAmpWorkspaceLayout,
  requirement_ordinal~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : JointDiffusionCudaAotLimits,
) -> JointDiffusionAudioVaeAmpStageCudaCandidate raise JointDiffusionCudaAotError {
  if compiler.reassociate() || compiler.fused_multiply_add() {
    raise Invalid(Compiler)
  }
  validate_request_shape(requirements, request)
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (
      AudioVaeDecoderAmpStage(
        stage,
        batch,
        channels,
        3,
        3,
        Kernels3_7_11Dilations1_3_5,
        Float32OriginalPrecomputedActivationPackedParametersWorkspaceOutput,
        BatchChannelTime
      ),
      AudioVaeDecoderAmpStageAot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let time = 5175
  let elements = checked_mul(
    checked_mul(batch.to_int64(), channels.to_int64()),
    time.to_int64(),
  )
  if stage != 1 ||
    batch != 2 ||
    channels != 256 ||
    request.audio_latent_frames() != 207 ||
    elements != 2649600L ||
    elements > limits.maximum_elements ||
    parameter_layout.regions().length() != 126 ||
    parameter_layout.total_arena_bytes() <= 0L ||
    workspace_layout.stages().length() != 97 {
    raise Invalid(Requirement)
  }
  validate_amp_parameter_geometry(parameter_layout, channels)
  @joint_audio_amp_workspace_layout.validate_amp_execution_stages(
    workspace_layout.stages(),
  ) catch {
    _ => raise Invalid(Requirement)
  }
  let expected_workspace = @joint_audio_amp_workspace_layout.amp_workspace_layout(
    element_count=elements,
    intermediate_element_count=checked_mul(elements, 2L),
    normalized_weight_element_count=720896L,
  ) catch {
    _ => raise Invalid(Requirement)
  }
  if expected_workspace.digest() != workspace_layout.digest() {
    raise Invalid(Requirement)
  }
  let stages : Array[JointDiffusionAudioVaeFirstAmpExecutionStage] = []
  for amp_stage in workspace_layout.stages() {
    let parameter_indices = match amp_stage {
      Activation(block, activation, phase) =>
        amp_activation_parameter_indices(
          parameter_layout, block, activation, phase,
        )
      Convolution(block, pair, side, phase) =>
        amp_convolution_parameter_indices(
          parameter_layout, block, pair, side, phase,
        )
      _ => []
    }
    let count = match amp_stage {
      Activation(_, _, Upsample) | Activation(_, _, SnakeBeta) =>
        checked_mul(elements, 2L)
      Convolution(_, _, _, Normalize) => channels.to_int64()
      _ => elements
    }
    stages.push({
      identity: amp_stage,
      function_symbol: amp_stage_symbol(amp_stage),
      dimensions: amp_stage_dimensions(
        count,
        if amp_stage is Convolution(_, _, _, Normalize) {
          1
        } else {
          256
        },
      ),
      parameter_region_indices: ReadOnlyArray::from_array(parameter_indices),
      workspace_roles: ReadOnlyArray::from_array(
        amp_stage_workspace_roles(amp_stage),
      ),
    })
  }
  if stages.length() != 97 {
    abort("validated AMP layout changed stage count")
  }
  let value_bytes = checked_mul(elements, 4L)
  let operands = [
    launch_operand(AudioDecoderUpsampleTransitionOutput, value_bytes, 16L),
    launch_operand(AudioDecoderAmpActivationOutput, value_bytes, 16L),
    launch_operand(ModelWeights, parameter_layout.total_arena_bytes(), 64L),
    launch_operand(Workspace, workspace_layout.workspace_bytes(), 256L),
    launch_operand(AudioDecoderAmpStageOutput, value_bytes, 16L),
  ]
  let source = render_amp_stage_source(batch, channels, time, elements)
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = JointDiffusionCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let identity = compiler.identity()
  let stage_text = StringBuilder()
  for ordinal, execution in stages {
    stage_text <+
      "|\{ordinal}:\{execution.function_symbol}:\{execution.dimensions.grid_x()}:\{execution.dimensions.block_x()}:p="
    for index in execution.parameter_region_indices {
      let region = parameter_layout.regions()[index]
      stage_text <+ "\{index}@\{region.byte_offset()}/\{region.byte_count()},"
    }
    stage_text.write_string("w=")
    for role in execution.workspace_roles {
      stage_text <+ "\{amp_workspace_role_tag(role)},"
    }
  }
  let recipe = @utf8.encode(
    "lunaflux.joint-diffusion-audio-vae-amp-stage.v1|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|request=\{request.height()}:\{request.width()}:\{request.frames()}:\{request.inference_steps()}:\{request.audio_latent_frames()}" +
    "|ordinal=\{requirement_ordinal}|stage=\{stage}|geometry=2:256:5175:3:3" +
    "|parameter-layout=" +
    parameter_layout.digest().as_hex() +
    "|workspace-layout=" +
    workspace_layout.digest().as_hex() +
    "|workspace-bytes=\{workspace_layout.workspace_bytes()}" +
    "|sm=\{target.compute_major()}.\{target.compute_minor()}:\{target.supports_bf16()}:\{target.supports_cublas_lt()}" +
    "|compiler=" +
    identity.toolchain_digest_sha256() +
    ":\{identity.major()}.\{identity.minor()}.\{identity.patch()}" +
    "|opt=\{compiler.optimization_level()}|fma=\{compiler.fused_multiply_add()}|reassociate=\{compiler.reassociate()}|max-registers=\{compiler.max_registers()}" +
    "|launches=30+33+33+1" +
    stage_text.to_string() +
    "|source=" +
    source_digest.as_hex(),
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    numerical_contract: OrderedF32AmpParallelResidualAverageV1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    request_shape: request,
    stage_index: stage,
    parameter_layout_digest: parameter_layout.digest(),
    workspace_layout_digest: workspace_layout.digest(),
    batch_count: batch,
    channel_count: channels,
    time_count: time,
    element_count: elements,
    target,
    operands: ReadOnlyArray::from_array(operands),
    stages: ReadOnlyArray::from_array(stages),
    compiler,
    source,
    source_digest,
    recipe,
    recipe_digest: JointDiffusionCudaRecipeDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(recipe)),
    ),
  }
}
