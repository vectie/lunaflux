///|
pub fn lower_audio_vae_decoder_amp_activation_cuda_aot_candidate(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  parameters~ : @joint_diffusion_vae_reference.JointVaeAudioDecoderAmpActivation,
  requirement_ordinal~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : JointDiffusionCudaAotLimits,
) -> JointDiffusionAudioVaeAmpActivationCudaCandidate raise JointDiffusionCudaAotError {
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
      AudioVaeDecoderAmpActivation(
        stage,
        batch,
        channels,
        ratio,
        kernel,
        storage,
        layout
      ),
      AudioVaeDecoderAmpActivationAot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let profile_matches = match stage {
    1 =>
      batch == 2 &&
      channels == 256 &&
      parameters.time_count() == 5175 &&
      parameters.upsample_time() == 10350 &&
      parameters.alpha_name() == "decoder.resblocks.3.activations.0.act.alpha" &&
      parameters.beta_name() == "decoder.resblocks.3.activations.0.act.beta" &&
      parameters.upsample_filter_name() ==
      "decoder.resblocks.3.activations.0.upsample.filter" &&
      parameters.downsample_filter_name() ==
      "decoder.resblocks.3.activations.0.downsample.lowpass.filter"
    2 =>
      batch == 2 &&
      channels == 128 &&
      parameters.time_count() == 10350 &&
      parameters.upsample_time() == 20700 &&
      parameters.alpha_name() == "decoder.resblocks.6.activations.0.act.alpha" &&
      parameters.beta_name() == "decoder.resblocks.6.activations.0.act.beta" &&
      parameters.upsample_filter_name() ==
      "decoder.resblocks.6.activations.0.upsample.filter" &&
      parameters.downsample_filter_name() ==
      "decoder.resblocks.6.activations.0.downsample.lowpass.filter"
    _ => false
  }
  if parameters.plan_digest() != requirements.source_identity().plan() ||
    stage != parameters.stage_index() ||
    batch != parameters.decoder_batch_count() ||
    channels != parameters.channel_count() ||
    ratio != parameters.ratio() ||
    kernel != parameters.kernel_size() ||
    storage != parameters.storage() ||
    layout != parameters.layout() ||
    !profile_matches ||
    ratio != 2 ||
    kernel != 12 ||
    storage !=
    Float32InputAlphaBetaUpsampleFilterDownsampleFilterWorkspacesOutput ||
    layout != BatchChannelTime {
    raise Invalid(Requirement)
  }
  let elements = parameters.element_count()
  let intermediate = parameters.intermediate_element_count()
  if elements != 2649600L ||
    intermediate != checked_mul(elements, 2L) ||
    intermediate > limits.maximum_elements {
    raise Invalid(Limit)
  }
  let element_bytes = checked_mul(elements, 4L)
  let intermediate_bytes = checked_mul(intermediate, 4L)
  let channel_bytes = checked_mul(channels.to_int64(), 4L)
  let filter_bytes = checked_mul(kernel.to_int64(), 4L)
  let operands = [
    launch_operand(
      if stage == 1 {
        AudioDecoderUpsampleTransitionOutput
      } else {
        AudioDecoderUpsampleTransition2Output
      },
      element_bytes,
      16L,
    ),
    launch_operand(AudioDecoderAmpAlpha, channel_bytes, 16L),
    launch_operand(AudioDecoderAmpBeta, channel_bytes, 16L),
    launch_operand(AudioDecoderAmpUpsampleFilter, filter_bytes, 16L),
    launch_operand(AudioDecoderAmpDownsampleFilter, filter_bytes, 16L),
    launch_operand(AudioDecoderAmpUpsampleWorkspace, intermediate_bytes, 16L),
    launch_operand(AudioDecoderAmpActivatedWorkspace, intermediate_bytes, 16L),
    launch_operand(
      if stage == 1 {
        AudioDecoderAmpActivationOutput
      } else {
        AudioDecoderAmpActivation2Output
      },
      element_bytes,
      16L,
    ),
  ]
  let upsample_symbol = "lunaflux_joint_diffusion_audio_vae_amp_\{stage}_upsample_v1"
  let activation_symbol = "lunaflux_joint_diffusion_audio_vae_amp_\{stage}_snake_beta_v1"
  let downsample_symbol = "lunaflux_joint_diffusion_audio_vae_amp_\{stage}_downsample_v1"
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
    abort("shared alias-free renderer violated AMP activation")
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
    "lunaflux.joint-diffusion-audio-vae-amp-activation.v1|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|parameters=" +
    parameters.digest().as_hex() +
    "|ordinal=\{requirement_ordinal}|stage=\{stage}|layout=b-c-t|storage=f32-input-alpha-beta-up-filter-down-filter-workspaces-output" +
    "|geometry=\{batch}:\{channels}:\{parameters.time_count()}:\{parameters.upsample_time()}:2:12" +
    "|weights=\{parameters.alpha_name()}:\{parameters.beta_name()}:\{parameters.upsample_filter_name()}:\{parameters.downsample_filter_name()}" +
    "|sm=\{target.compute_major()}.\{target.compute_minor()}:\{target.supports_bf16()}:\{target.supports_cublas_lt()}" +
    "|compiler=" +
    identity.toolchain_digest_sha256() +
    ":\{identity.major()}.\{identity.minor()}.\{identity.patch()}" +
    "|opt=\{compiler.optimization_level()}|fma=\{compiler.fused_multiply_add()}|reassociate=\{compiler.reassociate()}|max-registers=\{compiler.max_registers()}" +
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
    stage_index: stage,
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
