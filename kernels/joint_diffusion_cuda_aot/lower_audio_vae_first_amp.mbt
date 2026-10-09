///|
fn first_amp_dimensions(
  elements : Int64,
) -> @launch_contract.AotLaunchDimensions raise JointDiffusionCudaAotError {
  let grid = (elements + 255L) / 256L
  if grid > 2147483647L {
    raise Invalid(Limit)
  }
  @launch_contract.AotLaunchDimensions::new(
    grid_x=grid.to_int(),
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    error => raise Launch(error)
  }
}

///|
fn first_amp_stage_text(
  stages : ArrayView[JointDiffusionAudioVaeFirstAmpCudaStage],
) -> String {
  let text = StringBuilder()
  for index, stage in stages {
    if index > 0 {
      text.write_char(';')
    }
    text.write_string(
      match stage.kind {
        Upsample => "upsample"
        SnakeBeta => "snake-beta"
        Downsample => "downsample"
      },
    )
    text <+
      "@\{stage.function_symbol}:\{stage.dimensions.grid_x()}:\{stage.dimensions.block_x()}["
    for operand_index, operand in stage.operands {
      if operand_index > 0 {
        text.write_char(',')
      }
      text <+
        "\{role_tag(operand.role())}@\{operand.byte_count()}/\{operand.alignment()}"
    }
    text.write_char(']')
  }
  text.to_string()
}

///|
pub fn lower_audio_vae_decoder_first_amp_activation_cuda_aot_candidate(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  parameters~ : @joint_diffusion_vae_reference.JointVaeAudioDecoderFirstAmpActivation,
  requirement_ordinal~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : JointDiffusionCudaAotLimits,
) -> JointDiffusionAudioVaeFirstAmpCudaCandidate raise JointDiffusionCudaAotError {
  if compiler.reassociate() || compiler.fused_multiply_add() {
    raise Invalid(Compiler)
  }
  validate_request_shape(requirements, parameters.request_shape())
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (
      AudioVaeDecoderFirstAmpActivation(
        batch,
        channels,
        ratio,
        kernel,
        storage,
        layout
      ),
      AudioVaeDecoderFirstAmpActivationAot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  if parameters.plan_digest() != requirements.source_identity().plan() ||
    batch != parameters.decoder_batch_count() ||
    channels != parameters.channel_count() ||
    ratio != parameters.ratio() ||
    kernel != parameters.kernel_size() ||
    storage != parameters.storage() ||
    layout != parameters.layout() ||
    batch != 2 ||
    channels != 512 ||
    ratio != 2 ||
    kernel != 12 ||
    storage !=
    Float32InputAlphaBetaUpsampleFilterDownsampleFilterWorkspacesOutput ||
    layout != BatchChannelTime ||
    parameters.alpha_name() != "decoder.resblocks.0.activations.0.act.alpha" ||
    parameters.beta_name() != "decoder.resblocks.0.activations.0.act.beta" ||
    parameters.upsample_filter_name() !=
    "decoder.resblocks.0.activations.0.upsample.filter" ||
    parameters.downsample_filter_name() !=
    "decoder.resblocks.0.activations.0.downsample.lowpass.filter" {
    raise Invalid(Requirement)
  }
  let elements = parameters.element_count()
  let intermediate = parameters.intermediate_element_count()
  if elements <= 0L ||
    intermediate != checked_mul(elements, 2L) ||
    intermediate > limits.maximum_elements {
    raise Invalid(Limit)
  }
  let element_bytes = checked_mul(elements, 4L)
  let intermediate_bytes = checked_mul(intermediate, 4L)
  let channel_bytes = checked_mul(channels.to_int64(), 4L)
  let filter_bytes = checked_mul(kernel.to_int64(), 4L)
  let operands = [
    launch_operand(AudioDecoderFirstUpsampleOutput, element_bytes, 16L),
    launch_operand(AudioDecoderFirstAmpAlpha, channel_bytes, 16L),
    launch_operand(AudioDecoderFirstAmpBeta, channel_bytes, 16L),
    launch_operand(AudioDecoderFirstAmpUpsampleFilter, filter_bytes, 16L),
    launch_operand(AudioDecoderFirstAmpDownsampleFilter, filter_bytes, 16L),
    launch_operand(
      AudioDecoderFirstAmpUpsampleWorkspace,
      intermediate_bytes,
      16L,
    ),
    launch_operand(
      AudioDecoderFirstAmpActivatedWorkspace,
      intermediate_bytes,
      16L,
    ),
    launch_operand(AudioDecoderFirstAmpActivationOutput, element_bytes, 16L),
  ]
  let upsample_symbol = "lunaflux_joint_diffusion_audio_vae_first_amp_upsample_v1"
  let activation_symbol = "lunaflux_joint_diffusion_audio_vae_first_amp_snake_beta_v1"
  let downsample_symbol = "lunaflux_joint_diffusion_audio_vae_first_amp_downsample_v1"
  let stages : Array[JointDiffusionAudioVaeFirstAmpCudaStage] = [
    {
      kind: Upsample,
      function_symbol: upsample_symbol,
      dimensions: first_amp_dimensions(intermediate),
      operands: ReadOnlyArray::from_array([
        operands[0],
        operands[3],
        operands[5],
      ]),
    },
    {
      kind: SnakeBeta,
      function_symbol: activation_symbol,
      dimensions: first_amp_dimensions(intermediate),
      operands: ReadOnlyArray::from_array([
        operands[5],
        operands[1],
        operands[2],
        operands[6],
      ]),
    },
    {
      kind: Downsample,
      function_symbol: downsample_symbol,
      dimensions: first_amp_dimensions(elements),
      operands: ReadOnlyArray::from_array([
        operands[6],
        operands[4],
        operands[7],
      ]),
    },
  ]
  let rendered = @alias_free_activation_cuda_source.f32_alias_free_snake_beta_source(
    batch_count=batch,
    channel_count=channels,
    time_count=parameters.time_count(),
    ratio~,
    kernel_size=kernel,
    upsample_symbol~,
    activation_symbol~,
    downsample_symbol~,
  ) catch {
    error => raise AliasFreeSource(error)
  }
  if rendered.numerical_contract() != OrderedF32UpsampleSnakeBetaDownsampleV1 ||
    rendered.input_element_count() != elements ||
    rendered.intermediate_element_count() != intermediate ||
    rendered.output_element_count() != elements ||
    rendered.upsample_time() != parameters.upsample_time() {
    abort("shared alias-free renderer violated first AMP activation")
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = JointDiffusionCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let identity = compiler.identity()
  let recipe = @utf8.encode(
    "lunaflux.joint-diffusion-audio-vae-first-amp-activation.v1|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|parameters=" +
    parameters.digest().as_hex() +
    "|ordinal=" +
    requirement_ordinal.to_string() +
    "|layout=b-c-t|storage=f32-input-alpha-beta-up-filter-down-filter-workspaces-output|geometry=" +
    batch.to_string() +
    ":" +
    channels.to_string() +
    ":" +
    parameters.time_count().to_string() +
    ":" +
    parameters.upsample_time().to_string() +
    ":2:12" +
    "|weights=" +
    parameters.alpha_name() +
    ":" +
    parameters.beta_name() +
    ":" +
    parameters.upsample_filter_name() +
    ":" +
    parameters.downsample_filter_name() +
    "|sm=" +
    target.compute_major().to_string() +
    "." +
    target.compute_minor().to_string() +
    ":" +
    target.supports_bf16().to_string() +
    ":" +
    target.supports_cublas_lt().to_string() +
    "|compiler=" +
    identity.toolchain_digest_sha256() +
    ":" +
    identity.major().to_string() +
    "." +
    identity.minor().to_string() +
    "." +
    identity.patch().to_string() +
    "|opt=" +
    compiler.optimization_level().to_string() +
    "|fma=" +
    compiler.fused_multiply_add().to_string() +
    "|reassociate=" +
    compiler.reassociate().to_string() +
    "|max-registers=" +
    compiler.max_registers().to_string() +
    "|stages=" +
    first_amp_stage_text(stages) +
    "|source=" +
    source_digest.as_hex(),
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    numerical_contract: OrderedF32AliasFreeSnakeBetaV1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    parameter_digest: parameters.digest(),
    request_shape: parameters.request_shape(),
    batch_count: batch,
    channel_count: channels,
    time_count: parameters.time_count(),
    upsample_time: parameters.upsample_time(),
    element_count: elements,
    intermediate_element_count: intermediate,
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
