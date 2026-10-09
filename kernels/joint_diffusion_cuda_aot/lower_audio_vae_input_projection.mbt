///|
pub fn lower_audio_vae_decoder_input_projection_cuda_aot_candidate(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  parameters~ : @joint_diffusion_vae_reference.JointVaeAudioDecoderInputProjection,
  requirement_ordinal~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : JointDiffusionCudaAotLimits,
) -> JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate raise JointDiffusionCudaAotError {
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
      AudioVaeDecoderInputProjection(
        batch,
        input,
        output,
        kernel,
        storage,
        layout
      ),
      AudioVaeDecoderInputProjectionAot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  if parameters.plan_digest() != requirements.source_identity().plan() ||
    batch != parameters.decoder_batch_count() ||
    input != parameters.input_channels() ||
    output != parameters.output_channels() ||
    kernel != parameters.kernel_size() ||
    storage != parameters.storage() ||
    layout != parameters.layout() ||
    kernel != 1 ||
    storage != Float32InputWeightBiasOutput ||
    layout != BatchChannelTime ||
    parameters.weight_name() != "dec_in_proj.weight" ||
    parameters.bias_name() != "dec_in_proj.bias" {
    raise Invalid(Requirement)
  }
  let time = request.audio_latent_frames()
  let input_elements = checked_mul(
    checked_mul(batch.to_int64(), input.to_int64()),
    time.to_int64(),
  )
  let weight_elements = checked_mul(output.to_int64(), input.to_int64())
  let output_elements = checked_mul(
    checked_mul(batch.to_int64(), output.to_int64()),
    time.to_int64(),
  )
  if parameters.input_element_count() != input_elements ||
    parameters.weight_element_count() != weight_elements ||
    parameters.bias_element_count() != output.to_int64() ||
    parameters.output_element_count() != output_elements ||
    input_elements > limits.maximum_elements ||
    weight_elements > limits.maximum_elements ||
    output_elements > limits.maximum_elements {
    raise Invalid(Limit)
  }
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=((output_elements + 255L) / 256L).to_int(),
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    error => raise Launch(error)
  }
  let operands = [
    launch_operand(
      AudioDestandardizedLatentOutput,
      checked_mul(input_elements, 4L),
      16L,
    ),
    launch_operand(
      AudioDecoderInputProjectionWeight,
      checked_mul(weight_elements, 4L),
      16L,
    ),
    launch_operand(
      AudioDecoderInputProjectionBias,
      checked_mul(output.to_int64(), 4L),
      16L,
    ),
    launch_operand(
      AudioDecoderInputProjectionOutput,
      checked_mul(output_elements, 4L),
      16L,
    ),
  ]
  let symbol = "lunaflux_joint_diffusion_audio_vae_dec_in_proj_v1"
  let rendered = @channel_major_projection_cuda_source.f32_pointwise_conv1d_source(
    batch_count=batch,
    input_channels=input,
    output_channels=output,
    time_count=time,
    function_symbol=symbol,
  ) catch {
    error => raise ChannelMajorSource(error)
  }
  if rendered.operation() != F32PointwiseConv1d ||
    rendered.numerical_contract() != OrderedF32BiasAccumulationF32OutputV1 ||
    rendered.input_element_count() != input_elements ||
    rendered.output_element_count() != output_elements {
    abort("shared channel-major renderer violated decoder input projection")
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = JointDiffusionCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let operand_text = StringBuilder()
  for index, operand in operands {
    if index > 0 {
      operand_text.write_char(',')
    }
    operand_text.write_string(role_tag(operand.role()))
    operand_text <+ "@\{operand.byte_count()}/\{operand.alignment()}"
  }
  let compiler_identity = compiler.identity()
  let recipe = @utf8.encode(
    "lunaflux.joint-diffusion-audio-vae-decoder-input-projection.v1" +
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
    "|layout=b-c-t|storage=f32-input-weight-bias-output" +
    "|batch=" +
    batch.to_string() +
    "|input=" +
    input.to_string() +
    "|output=" +
    output.to_string() +
    "|time=" +
    time.to_string() +
    "|kernel=1|stride=1|padding=0" +
    "|weight=" +
    parameters.weight_name() +
    "|bias=" +
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
    "|launch=" +
    dimensions.grid_x().to_string() +
    ":" +
    dimensions.block_x().to_string() +
    ":" +
    dimensions.shared_memory_bytes().to_string() +
    "|operands=" +
    operand_text.to_string() +
    "|symbol=" +
    symbol +
    "|source=" +
    source_digest.as_hex(),
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    numerical_contract: OrderedF32ChannelMajorPointwiseProjectionV1,
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
    input_element_count: input_elements,
    weight_element_count: weight_elements,
    bias_element_count: output.to_int64(),
    output_element_count: output_elements,
    target,
    dimensions,
    operands: ReadOnlyArray::from_array(operands),
    compiler,
    function_symbol: symbol,
    source,
    source_digest,
    recipe,
    recipe_digest: JointDiffusionCudaRecipeDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(recipe)),
    ),
  }
}
