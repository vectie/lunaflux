///|
pub fn lower_video_vae_latent_destandardization_cuda_aot_candidate(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  parameters~ : @joint_diffusion_vae_reference.JointVaeLatentDestandardization,
  requirement_ordinal~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : JointDiffusionCudaAotLimits,
) -> JointDiffusionVideoVaeLatentDestandardizationCudaCandidate raise JointDiffusionCudaAotError {
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
      VideoVaeLatentDestandardization(channels, storage, layout),
      VideoVaeLatentDestandardizationAot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  if parameters.plan_digest() != requirements.source_identity().plan() ||
    channels != parameters.channel_count() ||
    parameters.batch_count() != 1 ||
    storage != parameters.storage() ||
    layout != parameters.layout() ||
    storage != Float32InputScaleMeanOutput ||
    layout != BatchChannelFrameHeightWidth {
    raise Invalid(Requirement)
  }
  let expected_per_channel = checked_mul(
    checked_mul(
      request.video_latent_frames().to_int64(),
      request.video_latent_height().to_int64(),
    ),
    request.video_latent_width().to_int64(),
  )
  let elements = checked_mul(channels.to_int64(), expected_per_channel)
  if parameters.elements_per_channel() != expected_per_channel ||
    parameters.element_count() != elements ||
    elements > limits.maximum_elements {
    raise Invalid(Limit)
  }
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=((elements + 255L) / 256L).to_int(),
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    error => raise Launch(error)
  }
  let tensor_bytes = checked_mul(elements, 4L)
  let coefficient_bytes = checked_mul(channels.to_int64(), 4L)
  let operands = [
    launch_operand(VideoNormalizedLatentInput, tensor_bytes, 16L),
    launch_operand(VideoLatentStandardDeviationInput, coefficient_bytes, 16L),
    launch_operand(VideoLatentMeanInput, coefficient_bytes, 16L),
    launch_operand(VideoDestandardizedLatentOutput, tensor_bytes, 16L),
  ]
  let symbol = "lunaflux_joint_diffusion_video_vae_latent_destandardize_v1"
  let rendered = @affine_modulation_cuda_source.f32_channel_major_destandardization_source(
    batch_count=1,
    channels~,
    elements_per_channel=expected_per_channel,
    function_symbol=symbol,
  ) catch {
    error => raise AffineSource(error)
  }
  if rendered.operation() != F32ChannelMajorDestandardization ||
    rendered.numerical_contract() != OrderedF32ChannelMajorMultiplyAddV1 {
    abort("shared affine renderer violated video latent contract")
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
    "lunaflux.joint-diffusion-video-vae-latent-destandardization.v1" +
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
    "|channels=" +
    channels.to_string() +
    "|batch=1" +
    "|elements-per-channel=" +
    expected_per_channel.to_string() +
    "|elements=" +
    elements.to_string() +
    "|request=" +
    request.frames().to_string() +
    ":" +
    request.video_latent_frames().to_string() +
    ":" +
    request.video_latent_height().to_string() +
    ":" +
    request.video_latent_width().to_string() +
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
    numerical_contract: OrderedF32ChannelMajorDestandardizationV1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    parameter_digest: parameters.digest(),
    request_shape: request,
    batch_count: 1,
    channel_count: channels,
    elements_per_channel: expected_per_channel,
    element_count: elements,
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
