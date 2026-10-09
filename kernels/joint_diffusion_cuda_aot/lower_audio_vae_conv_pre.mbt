///|
fn conv_pre_dimensions(
  elements : Int64,
  block_x : Int,
) -> @launch_contract.AotLaunchDimensions raise JointDiffusionCudaAotError {
  let grid = (elements + block_x.to_int64() - 1L) / block_x.to_int64()
  if grid > 2147483647L {
    raise Invalid(Limit)
  }
  @launch_contract.AotLaunchDimensions::new(
    grid_x=grid.to_int(),
    grid_y=1,
    grid_z=1,
    block_x~,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    error => raise Launch(error)
  }
}

///|
fn conv_pre_stage_text(
  stages : ArrayView[JointDiffusionAudioVaeConvPreCudaStage],
) -> String {
  let text = StringBuilder()
  for stage_index, stage in stages {
    if stage_index > 0 {
      text.write_char(';')
    }
    text.write_string(
      match stage.kind {
        NormalizeWeight => "normalize"
        Convolve => "convolve"
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
pub fn lower_audio_vae_decoder_conv_pre_cuda_aot_candidate(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  parameters~ : @joint_diffusion_vae_reference.JointVaeAudioDecoderConvPre,
  requirement_ordinal~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : JointDiffusionCudaAotLimits,
) -> JointDiffusionAudioVaeConvPreCudaCandidate raise JointDiffusionCudaAotError {
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
      AudioVaeDecoderConvPre(
        batch,
        input,
        output,
        kernel,
        padding,
        stride,
        storage,
        axes,
        layout
      ),
      AudioVaeDecoderConvPreAot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  if parameters.plan_digest() != requirements.source_identity().plan() ||
    batch != parameters.decoder_batch_count() ||
    input != parameters.input_channels() ||
    output != parameters.output_channels() ||
    kernel != parameters.kernel_size() ||
    padding != parameters.padding() ||
    stride != parameters.stride() ||
    storage != parameters.storage() ||
    axes != parameters.axes() ||
    layout != parameters.layout() ||
    kernel != 7 ||
    padding != 3 ||
    stride != 1 ||
    storage != Float32InputWeightGWeightVBiasWorkspaceOutput ||
    axes != PerOutputChannelInputAndKernel ||
    layout != BatchChannelTime ||
    parameters.weight_g_name() != "decoder.conv_pre.weight_g" ||
    parameters.weight_v_name() != "decoder.conv_pre.weight_v" ||
    parameters.bias_name() != "decoder.conv_pre.bias" {
    raise Invalid(Requirement)
  }
  let time = request.audio_latent_frames()
  let input_elements = checked_mul(
    checked_mul(batch.to_int64(), input.to_int64()),
    time.to_int64(),
  )
  let weight_g_elements = output.to_int64()
  let weight_v_elements = checked_mul(
    checked_mul(output.to_int64(), input.to_int64()),
    kernel.to_int64(),
  )
  let output_elements = checked_mul(
    checked_mul(batch.to_int64(), output.to_int64()),
    time.to_int64(),
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
  let operands = [
    launch_operand(AudioDecoderInputProjectionOutput, input_bytes, 16L),
    launch_operand(AudioDecoderConvPreWeightG, weight_g_bytes, 16L),
    launch_operand(AudioDecoderConvPreWeightV, weight_v_bytes, 16L),
    launch_operand(AudioDecoderConvPreBias, bias_bytes, 16L),
    launch_operand(
      AudioDecoderConvPreNormalizedWeightWorkspace,
      weight_v_bytes,
      16L,
    ),
    launch_operand(AudioDecoderConvPreOutput, output_bytes, 16L),
  ]
  let normalize_symbol = "lunaflux_joint_diffusion_audio_vae_conv_pre_normalize_v1"
  let convolution_symbol = "lunaflux_joint_diffusion_audio_vae_conv_pre_execute_v1"
  let normalize_dimensions = conv_pre_dimensions(output.to_int64(), 1)
  let convolution_dimensions = conv_pre_dimensions(output_elements, 256)
  let stages : Array[JointDiffusionAudioVaeConvPreCudaStage] = [
    {
      kind: NormalizeWeight,
      function_symbol: normalize_symbol,
      dimensions: normalize_dimensions,
      operands: ReadOnlyArray::from_array([
        operands[1],
        operands[2],
        operands[4],
      ]),
    },
    {
      kind: Convolve,
      function_symbol: convolution_symbol,
      dimensions: convolution_dimensions,
      operands: ReadOnlyArray::from_array([
        operands[0],
        operands[4],
        operands[3],
        operands[5],
      ]),
    },
  ]
  let rendered = @channel_major_projection_cuda_source.f32_weight_normalized_conv1d_source(
    batch_count=batch,
    input_channels=input,
    output_channels=output,
    input_time=time,
    kernel_size=kernel,
    padding~,
    stride~,
    normalize_symbol~,
    convolution_symbol~,
  ) catch {
    error => raise ChannelMajorSource(error)
  }
  if rendered.operation() != F32WeightNormalizedConv1d ||
    rendered.numerical_contract() != OrderedF32PerOutputWeightNormConv1dV1 ||
    rendered.input_element_count() != input_elements ||
    rendered.weight_element_count() != weight_v_elements ||
    rendered.output_element_count() != output_elements ||
    rendered.output_time() != time {
    abort("shared channel-major renderer violated decoder conv_pre")
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
    "lunaflux.joint-diffusion-audio-vae-decoder-conv-pre.v1" +
    "|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|parameters=" +
    parameters.digest().as_hex() +
    "|ordinal=" +
    requirement_ordinal.to_string() +
    "|layout=b-c-t|storage=f32-input-weight-g-weight-v-bias-workspace-output" +
    "|weight-norm=per-output-channel-input-and-kernel" +
    "|geometry=" +
    batch.to_string() +
    ":" +
    input.to_string() +
    ":" +
    output.to_string() +
    ":" +
    time.to_string() +
    ":" +
    kernel.to_string() +
    ":" +
    padding.to_string() +
    ":" +
    stride.to_string() +
    "|weights=" +
    parameters.weight_g_name() +
    ":" +
    parameters.weight_v_name() +
    ":" +
    parameters.bias_name() +
    "|sm=" +
    target.compute_major().to_string() +
    "." +
    target.compute_minor().to_string() +
    ":" +
    target.supports_bf16().to_string() +
    ":" +
    target.supports_cublas_lt().to_string() +
    "|compiler=" +
    compiler_identity.toolchain_digest_sha256() +
    ":" +
    compiler_identity.major().to_string() +
    "." +
    compiler_identity.minor().to_string() +
    "." +
    compiler_identity.patch().to_string() +
    "|opt=" +
    compiler.optimization_level().to_string() +
    "|fma=" +
    compiler.fused_multiply_add().to_string() +
    "|reassociate=" +
    compiler.reassociate().to_string() +
    "|max-registers=" +
    compiler.max_registers().to_string() +
    "|stages=" +
    conv_pre_stage_text(stages) +
    "|source=" +
    source_digest.as_hex(),
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    numerical_contract: OrderedF32PerOutputWeightNormConv1dV1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    parameter_digest: parameters.digest(),
    request_shape: request,
    decoder_batch_count: batch,
    input_channels: input,
    output_channels: output,
    time_count: time,
    kernel_size: kernel,
    padding,
    stride,
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
