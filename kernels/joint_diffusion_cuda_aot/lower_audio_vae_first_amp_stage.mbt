///|
fn amp_stage_dimensions(
  count : Int64,
  block_size : Int,
) -> @launch_contract.AotLaunchDimensions raise JointDiffusionCudaAotError {
  let grid = (count + block_size.to_int64() - 1L) / block_size.to_int64()
  if grid <= 0L || grid > 2147483647L {
    raise Invalid(Limit)
  }
  @launch_contract.AotLaunchDimensions::new(
    grid_x=grid.to_int(),
    grid_y=1,
    grid_z=1,
    block_x=block_size,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    error => raise Launch(error)
  }
}

///|
fn amp_parameter_index(
  layout : @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayout,
  role : @joint_audio_amp_parameter_layout.JointAudioAmpParameterRole,
) -> Int raise JointDiffusionCudaAotError {
  for index, region in layout.regions() {
    if region.role() == role {
      return index
    }
  }
  raise Invalid(Requirement)
}

///|
fn amp_activation_parameter_indices(
  layout : @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayout,
  block : Int,
  activation : Int,
  phase : @joint_audio_amp_workspace_layout.JointAudioAmpActivationPhase,
) -> Array[Int] raise JointDiffusionCudaAotError {
  match phase {
    Upsample =>
      [amp_parameter_index(layout, ActivationUpsampleFilter(block, activation))]
    SnakeBeta =>
      [
        amp_parameter_index(layout, ActivationAlpha(block, activation)),
        amp_parameter_index(layout, ActivationBeta(block, activation)),
      ]
    Downsample =>
      [
        amp_parameter_index(
          layout,
          ActivationDownsampleFilter(block, activation),
        ),
      ]
  }
}

///|
fn amp_convolution_parameter_indices(
  layout : @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayout,
  block : Int,
  pair : Int,
  side : Int,
  phase : @joint_audio_amp_workspace_layout.JointAudioAmpConvolutionPhase,
) -> Array[Int] raise JointDiffusionCudaAotError {
  match phase {
    Normalize =>
      [
        amp_parameter_index(layout, ConvolutionWeightG(block, pair, side)),
        amp_parameter_index(layout, ConvolutionWeightV(block, pair, side)),
      ]
    Convolve =>
      [amp_parameter_index(layout, ConvolutionBias(block, pair, side))]
  }
}

///|
fn amp_state_role(
  _block : Int,
  pair : Int,
) -> @joint_audio_amp_workspace_layout.JointAudioAmpWorkspaceRole {
  if pair == 0 {
    OriginalUpsamplerOutput
  } else if pair == 1 {
    ResidualPingWorkspace
  } else {
    ResidualPongWorkspace
  }
}

///|
fn amp_residual_output_role(
  block : Int,
  pair : Int,
) -> @joint_audio_amp_workspace_layout.JointAudioAmpWorkspaceRole {
  if pair == 0 {
    ResidualPingWorkspace
  } else if pair == 1 {
    ResidualPongWorkspace
  } else {
    ParallelBlockOutput(block)
  }
}

///|
fn amp_stage_workspace_roles(
  stage : @joint_audio_amp_workspace_layout.JointAudioAmpExecutionStage,
) -> Array[@joint_audio_amp_workspace_layout.JointAudioAmpWorkspaceRole] {
  match stage {
    Activation(block, activation, Upsample) => {
      let pair = activation / 2
      [amp_state_role(block, pair), ActivationUpsampleWorkspace]
    }
    Activation(_, _, SnakeBeta) =>
      [ActivationUpsampleWorkspace, ActivationValueWorkspace]
    Activation(_, _, Downsample) =>
      [ActivationValueWorkspace, ActivationOutputWorkspace]
    Convolution(_, _, 0, Normalize) => [NormalizedWeightWorkspace]
    Convolution(block, pair, 0, Convolve) => {
      let input : @joint_audio_amp_workspace_layout.JointAudioAmpWorkspaceRole = if block ==
        0 &&
        pair == 0 {
        PrecomputedBlock0Activation0
      } else {
        ActivationOutputWorkspace
      }
      [input, NormalizedWeightWorkspace, ConvolutionOutputWorkspace]
    }
    Convolution(_, _, 1, Normalize) => [NormalizedWeightWorkspace]
    Convolution(_, _, 1, Convolve) =>
      [
        ActivationOutputWorkspace,
        NormalizedWeightWorkspace,
        ConvolutionOutputWorkspace,
      ]
    Residual(block, pair) =>
      [
        amp_state_role(block, pair),
        ConvolutionOutputWorkspace,
        amp_residual_output_role(block, pair),
      ]
    ParallelAverage =>
      [
        ParallelBlockOutput(0),
        ParallelBlockOutput(1),
        ParallelBlockOutput(2),
        FinalOutput,
      ]
    _ => []
  }
}

///|
fn amp_stage_symbol(
  stage : @joint_audio_amp_workspace_layout.JointAudioAmpExecutionStage,
) -> String {
  match stage {
    Activation(_, _, Upsample) => "lunaflux_audio_amp_activation_upsample_v1"
    Activation(_, _, SnakeBeta) => "lunaflux_audio_amp_activation_snake_beta_v1"
    Activation(_, _, Downsample) =>
      "lunaflux_audio_amp_activation_downsample_v1"
    Convolution(block, pair, side, phase) => {
      let kernel = [3, 7, 11][block]
      let dilation = if side == 0 { [1, 3, 5][pair] } else { 1 }
      let suffix = match phase {
        Normalize => "normalize"
        Convolve => "convolve"
      }
      "lunaflux_audio_amp_k\{kernel}_d\{dilation}_\{suffix}_v1"
    }
    Residual(_, _) => "lunaflux_audio_amp_residual_v1"
    ParallelAverage => "lunaflux_audio_amp_average_v1"
  }
}

///|
fn amp_workspace_role_tag(
  role : @joint_audio_amp_workspace_layout.JointAudioAmpWorkspaceRole,
) -> String {
  match role {
    OriginalUpsamplerOutput => "original"
    PrecomputedBlock0Activation0 => "precomputed-activation0"
    ActivationUpsampleWorkspace => "activation-upsample"
    ActivationValueWorkspace => "activation-value"
    ActivationOutputWorkspace => "activation-output"
    NormalizedWeightWorkspace => "normalized-weight"
    ConvolutionOutputWorkspace => "convolution-output"
    ResidualPingWorkspace => "residual-ping"
    ResidualPongWorkspace => "residual-pong"
    ParallelBlockOutput(block) => "block-output-\{block}"
    FinalOutput => "final-output"
  }
}

///|
fn append_amp_source(builder : StringBuilder, source : Bytes) -> Unit {
  builder.write_string(
    @utf8.decode(source) catch {
      _ => abort("renderer produced invalid UTF-8")
    },
  )
}

///|
fn render_amp_stage_source(
  batch : Int,
  channels : Int,
  time : Int,
  elements : Int64,
) -> Bytes raise JointDiffusionCudaAotError {
  let builder = StringBuilder()
  let activation = @alias_free_activation_cuda_source.f32_alias_free_snake_beta_source(
    batch_count=batch,
    channel_count=channels,
    time_count=time,
    ratio=2,
    kernel_size=12,
    upsample_symbol="lunaflux_audio_amp_activation_upsample_v1",
    activation_symbol="lunaflux_audio_amp_activation_snake_beta_v1",
    downsample_symbol="lunaflux_audio_amp_activation_downsample_v1",
  ) catch {
    error => raise AliasFreeSource(error)
  }
  append_amp_source(builder, activation.source_bytes())
  for block in 0..<3 {
    let kernel = [3, 7, 11][block]
    for dilation in [1, 3, 5] {
      let padding = dilation * (kernel - 1) / 2
      let rendered = @channel_major_projection_cuda_source.f32_weight_normalized_dilated_conv1d_source(
        batch_count=batch,
        channel_count=channels,
        time_count=time,
        kernel_size=kernel,
        dilation~,
        padding~,
        normalize_symbol="lunaflux_audio_amp_k\{kernel}_d\{dilation}_normalize_v1",
        convolution_symbol="lunaflux_audio_amp_k\{kernel}_d\{dilation}_convolve_v1",
      ) catch {
        error => raise ChannelMajorSource(error)
      }
      append_amp_source(builder, rendered.source_bytes())
    }
  }
  let residual = @joint_audio_amp_cuda_source.f32_residual_and_three_way_average_source(
    element_count=elements,
    residual_symbol="lunaflux_audio_amp_residual_v1",
    average_symbol="lunaflux_audio_amp_average_v1",
  ) catch {
    error => raise JointAudioAmpSource(error)
  }
  append_amp_source(builder, residual.source_bytes())
  @utf8.encode(builder.to_string())
}

///|
pub fn lower_audio_vae_decoder_first_amp_stage_cuda_aot_candidate(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  request~ : @joint_diffusion_plan.JointDiffusionRequestShape,
  parameter_layout~ : @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayout,
  workspace_layout~ : @joint_audio_amp_workspace_layout.JointAudioAmpWorkspaceLayout,
  requirement_ordinal~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : JointDiffusionCudaAotLimits,
) -> JointDiffusionAudioVaeFirstAmpStageCudaCandidate raise JointDiffusionCudaAotError {
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
      AudioVaeDecoderFirstAmpStage(
        batch,
        channels,
        3,
        3,
        Float32OriginalPrecomputedActivationPackedParametersWorkspaceOutput,
        BatchChannelTime
      ),
      AudioVaeDecoderFirstAmpStageAot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let time = request.audio_latent_frames() * 5
  let elements = checked_mul(
    checked_mul(batch.to_int64(), channels.to_int64()),
    time.to_int64(),
  )
  if batch != 2 ||
    channels != 512 ||
    time != 1035 ||
    elements > limits.maximum_elements ||
    parameter_layout.regions().length() != 126 ||
    parameter_layout.total_arena_bytes() != 132270336L ||
    workspace_layout.stages().length() != 97 {
    raise Invalid(Requirement)
  }
  validate_amp_parameter_geometry(parameter_layout, channels)
  @joint_audio_amp_workspace_layout.validate_first_stage_amp_execution_stages(
    workspace_layout.stages(),
  ) catch {
    _ => raise Invalid(Requirement)
  }
  let expected_workspace = @joint_audio_amp_workspace_layout.first_stage_amp_workspace_layout(
    element_count=elements,
    intermediate_element_count=checked_mul(elements, 2L),
    normalized_weight_element_count=2883584L,
  ) catch {
    _ => raise Invalid(Requirement)
  }
  if expected_workspace.digest() != workspace_layout.digest() {
    raise Invalid(Requirement)
  }
  let stages : Array[JointDiffusionAudioVaeFirstAmpExecutionStage] = []
  for stage in workspace_layout.stages() {
    let parameter_indices = match stage {
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
    let count = match stage {
      Activation(_, _, Upsample) | Activation(_, _, SnakeBeta) =>
        checked_mul(elements, 2L)
      Convolution(_, _, _, Normalize) => channels.to_int64()
      _ => elements
    }
    stages.push({
      identity: stage,
      function_symbol: amp_stage_symbol(stage),
      dimensions: amp_stage_dimensions(
        count,
        if stage is Convolution(_, _, _, Normalize) {
          1
        } else {
          256
        },
      ),
      parameter_region_indices: ReadOnlyArray::from_array(parameter_indices),
      workspace_roles: ReadOnlyArray::from_array(
        amp_stage_workspace_roles(stage),
      ),
    })
  }
  if stages.length() != 97 {
    abort("validated layout changed stage count")
  }
  let value_bytes = checked_mul(elements, 4L)
  let operands = [
    launch_operand(AudioDecoderFirstUpsampleOutput, value_bytes, 16L),
    launch_operand(AudioDecoderFirstAmpActivationOutput, value_bytes, 16L),
    launch_operand(ModelWeights, parameter_layout.total_arena_bytes(), 64L),
    launch_operand(Workspace, workspace_layout.workspace_bytes(), 256L),
    launch_operand(AudioDecoderFirstAmpStageOutput, value_bytes, 16L),
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
  for ordinal, stage in stages {
    stage_text <+
      "|\{ordinal}:\{stage.function_symbol}:\{stage.dimensions.grid_x()}:\{stage.dimensions.block_x()}:p="
    for index in stage.parameter_region_indices {
      let region = parameter_layout.regions()[index]
      stage_text <+ "\{index}@\{region.byte_offset()}/\{region.byte_count()},"
    }
    stage_text.write_string("w=")
    for role in stage.workspace_roles {
      stage_text <+ "\{amp_workspace_role_tag(role)},"
    }
  }
  let recipe = @utf8.encode(
    "lunaflux.joint-diffusion-audio-vae-first-amp-stage.v1|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|request=\{request.height()}:\{request.width()}:\{request.frames()}:\{request.inference_steps()}:\{request.audio_latent_frames()}" +
    "|ordinal=\{requirement_ordinal}|geometry=2:512:1035:3:3|parameter-layout=" +
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
