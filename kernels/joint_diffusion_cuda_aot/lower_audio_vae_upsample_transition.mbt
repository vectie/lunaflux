///|
pub fn lower_audio_vae_decoder_upsample_transition_cuda_aot_candidate(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  parameters~ : @joint_diffusion_vae_reference.JointVaeAudioDecoderUpsampleTransition,
  requirement_ordinal~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : JointDiffusionCudaAotLimits,
) -> JointDiffusionAudioVaeUpsampleTransitionCudaCandidate raise JointDiffusionCudaAotError {
  if compiler.reassociate() || compiler.fused_multiply_add() {
    raise Invalid(Compiler)
  }
  let request = parameters.request_shape()
  validate_request_shape(requirements, request)
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (
      AudioVaeDecoderUpsampleTransition(
        stage,
        batch,
        input,
        output,
        kernel,
        stride,
        padding,
        storage,
        axes,
        layout
      ),
      AudioVaeDecoderUpsampleTransitionAot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let profile_matches = match stage {
    1 =>
      batch == 2 &&
      input == 512 &&
      output == 256 &&
      parameters.input_time() == 1035 &&
      parameters.output_time() == 5175 &&
      kernel == 9 &&
      stride == 5 &&
      padding == 2 &&
      parameters.weight_g_name() == "decoder.ups.1.0.weight_g" &&
      parameters.weight_v_name() == "decoder.ups.1.0.weight_v" &&
      parameters.bias_name() == "decoder.ups.1.0.bias"
    2 =>
      batch == 2 &&
      input == 256 &&
      output == 128 &&
      parameters.input_time() == 5175 &&
      parameters.output_time() == 10350 &&
      kernel == 4 &&
      stride == 2 &&
      padding == 1 &&
      parameters.weight_g_name() == "decoder.ups.2.0.weight_g" &&
      parameters.weight_v_name() == "decoder.ups.2.0.weight_v" &&
      parameters.bias_name() == "decoder.ups.2.0.bias"
    _ => false
  }
  if parameters.plan_digest() != requirements.source_identity().plan() ||
    stage != parameters.stage_index() ||
    batch != parameters.decoder_batch_count() ||
    input != parameters.input_channels() ||
    output != parameters.output_channels() ||
    kernel != parameters.kernel_size() ||
    stride != parameters.stride() ||
    padding != parameters.padding() ||
    storage != parameters.storage() ||
    axes != parameters.axes() ||
    layout != parameters.layout() ||
    !profile_matches ||
    storage != Float32InputWeightGWeightVBiasWorkspaceOutput ||
    axes != PerInputChannelOutputAndKernel ||
    layout != BatchChannelTime {
    raise Invalid(Requirement)
  }
  let input_elements = checked_mul(
    checked_mul(batch.to_int64(), input.to_int64()),
    parameters.input_time().to_int64(),
  )
  let weight_g_elements = input.to_int64()
  let weight_v_elements = checked_mul(
    checked_mul(input.to_int64(), output.to_int64()),
    kernel.to_int64(),
  )
  let output_elements = checked_mul(
    checked_mul(batch.to_int64(), output.to_int64()),
    parameters.output_time().to_int64(),
  )
  if parameters.input_element_count() != input_elements ||
    parameters.weight_g_element_count() != weight_g_elements ||
    parameters.weight_v_element_count() != weight_v_elements ||
    parameters.bias_element_count() != output.to_int64() ||
    parameters.output_element_count() != output_elements ||
    input_elements > limits.maximum_elements ||
    weight_v_elements > limits.maximum_elements ||
    output_elements > limits.maximum_elements {
    raise Invalid(Limit)
  }
  let input_bytes = checked_mul(input_elements, 4L)
  let weight_g_bytes = checked_mul(weight_g_elements, 4L)
  let weight_v_bytes = checked_mul(weight_v_elements, 4L)
  let bias_bytes = checked_mul(output.to_int64(), 4L)
  let output_bytes = checked_mul(output_elements, 4L)
  let operands = if stage == 1 {
    [
      launch_operand(AudioDecoderFirstAmpStageOutput, input_bytes, 16L),
      launch_operand(AudioDecoderUpsampleTransitionWeightG, weight_g_bytes, 16L),
      launch_operand(AudioDecoderUpsampleTransitionWeightV, weight_v_bytes, 16L),
      launch_operand(AudioDecoderUpsampleTransitionBias, bias_bytes, 16L),
      launch_operand(
        AudioDecoderUpsampleTransitionNormalizedWeightWorkspace,
        weight_v_bytes,
        16L,
      ),
      launch_operand(AudioDecoderUpsampleTransitionOutput, output_bytes, 16L),
    ]
  } else {
    [
      launch_operand(AudioDecoderAmpStageOutput, input_bytes, 16L),
      launch_operand(
        AudioDecoderUpsampleTransition2WeightG,
        weight_g_bytes,
        16L,
      ),
      launch_operand(
        AudioDecoderUpsampleTransition2WeightV,
        weight_v_bytes,
        16L,
      ),
      launch_operand(AudioDecoderUpsampleTransition2Bias, bias_bytes, 16L),
      launch_operand(
        AudioDecoderUpsampleTransition2NormalizedWeightWorkspace,
        weight_v_bytes,
        16L,
      ),
      launch_operand(AudioDecoderUpsampleTransition2Output, output_bytes, 16L),
    ]
  }
  let normalize_symbol = "lunaflux_joint_diffusion_audio_vae_upsample_\{stage}_normalize_v1"
  let convolution_symbol = "lunaflux_joint_diffusion_audio_vae_upsample_\{stage}_execute_v1"
  let stages : Array[JointDiffusionAudioVaeFirstUpsampleCudaStage] = [
    {
      kind: NormalizeWeight,
      function_symbol: normalize_symbol,
      dimensions: first_upsample_dimensions(input.to_int64(), 1),
      operands: ReadOnlyArray::from_array([
        operands[1],
        operands[2],
        operands[4],
      ]),
    },
    {
      kind: ConvolveTranspose,
      function_symbol: convolution_symbol,
      dimensions: first_upsample_dimensions(output_elements, 256),
      operands: ReadOnlyArray::from_array([
        operands[0],
        operands[4],
        operands[3],
        operands[5],
      ]),
    },
  ]
  let rendered = @channel_major_projection_cuda_source.f32_weight_normalized_conv_transpose1d_source(
    batch_count=batch,
    input_channels=input,
    output_channels=output,
    input_time=parameters.input_time(),
    kernel_size=kernel,
    padding~,
    stride~,
    normalize_symbol~,
    convolution_symbol~,
  ) catch {
    error => raise ChannelMajorSource(error)
  }
  if rendered.operation() != F32WeightNormalizedConvTranspose1d ||
    rendered.numerical_contract() !=
    OrderedF32PerInputWeightNormConvTranspose1dV1 ||
    rendered.input_element_count() != input_elements ||
    rendered.weight_element_count() != weight_v_elements ||
    rendered.output_element_count() != output_elements ||
    rendered.output_time() != parameters.output_time() {
    abort("shared renderer violated decoder upsample transition")
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = JointDiffusionCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let compiler_identity = compiler.identity()
  let recipe = @utf8.encode(
    "lunaflux.joint-diffusion-audio-vae-upsample-transition.v1|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|parameters=" +
    parameters.digest().as_hex() +
    "|ordinal=\{requirement_ordinal}|stage=\{stage}|layout=b-c-t|storage=f32-input-weight-g-weight-v-bias-workspace-output" +
    "|weight-norm=per-input-channel-output-and-kernel|geometry=\{batch}:\{input}:\{output}:\{parameters.input_time()}:\{parameters.output_time()}:\{kernel}:\{stride}:\{padding}" +
    "|weights=\{parameters.weight_g_name()}:\{parameters.weight_v_name()}:\{parameters.bias_name()}" +
    "|sm=\{target.compute_major()}.\{target.compute_minor()}:\{target.supports_bf16()}:\{target.supports_cublas_lt()}" +
    "|compiler=" +
    compiler_identity.toolchain_digest_sha256() +
    ":\{compiler_identity.major()}.\{compiler_identity.minor()}.\{compiler_identity.patch()}" +
    "|opt=\{compiler.optimization_level()}|fma=\{compiler.fused_multiply_add()}|reassociate=\{compiler.reassociate()}|max-registers=\{compiler.max_registers()}" +
    "|stages=" +
    first_upsample_stage_text(stages) +
    "|source=" +
    source_digest.as_hex(),
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    numerical_contract: OrderedF32PerInputWeightNormConvTranspose1dV1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    parameter_digest: parameters.digest(),
    request_shape: request,
    stage_index: stage,
    decoder_batch_count: batch,
    input_channels: input,
    output_channels: output,
    input_time: parameters.input_time(),
    output_time: parameters.output_time(),
    kernel_size: kernel,
    stride,
    padding,
    input_element_count: input_elements,
    weight_g_element_count: weight_g_elements,
    weight_v_element_count: weight_v_elements,
    bias_element_count: output.to_int64(),
    output_element_count: output_elements,
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
