///|
fn output_projection_recipe(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  ordinal : Int,
  latent : @joint_diffusion_plan.JointDiffusionLatentKind,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
  rows : Int64,
  input_width : Int,
  output_width : Int,
  input_elements : Int64,
  output_elements : Int64,
  weight_elements : Int64,
  bias_elements : Int64,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  dimensions : @launch_contract.AotLaunchDimensions,
  operands : ArrayView[
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
  ],
  source_digest : JointDiffusionCudaSourceDigest,
) -> Bytes {
  let compiler_identity = compiler.identity()
  let operand_text = StringBuilder()
  for index, item in operands {
    if index > 0 {
      operand_text.write_char(',')
    }
    operand_text.write_string(role_tag(item.role()))
    operand_text.write_char('@')
    operand_text.write_string(item.byte_count().to_string())
    operand_text.write_char('/')
    operand_text.write_string(item.alignment().to_string())
  }
  let latent_tag = match latent {
    VideoLatent => "video"
    AudioLatent => "audio"
  }
  @utf8.encode(
    "lunaflux.joint-diffusion-cuda-aot.v1" +
    "|numerical=ordered-f32-dense-f32-output-v1" +
    "|storage=f32-selected-hidden,row-major-output-input-f32-weight,separate-f32-bias,f32-output" +
    "|selection=target-modality-before-row-independent-dense" +
    "|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|ordinal=" +
    ordinal.to_string() +
    "|latent=" +
    latent_tag +
    "|request=" +
    request.height().to_string() +
    "x" +
    request.width().to_string() +
    ":" +
    request.frames().to_string() +
    ":" +
    request.inference_steps().to_string() +
    ":" +
    request.video_latent_frames().to_string() +
    ":" +
    request.video_latent_height().to_string() +
    ":" +
    request.video_latent_width().to_string() +
    ":" +
    request.audio_latent_frames().to_string() +
    "|dense=" +
    rows.to_string() +
    ":" +
    input_width.to_string() +
    ":" +
    output_width.to_string() +
    ":" +
    input_elements.to_string() +
    ":" +
    output_elements.to_string() +
    ":" +
    weight_elements.to_string() +
    ":" +
    bias_elements.to_string() +
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
    "|operands=" +
    operand_text.to_string() +
    "|source=" +
    source_digest.as_hex(),
  )
}

///|
/// Lowers one exact video or audio output head. The caller supplies only the
/// target-modality rows after final normalization and F32 conversion; because
/// dense projection is row independent, this is numerically identical to the
/// official project-all-rows then index-select order for retained rows.
pub fn lower_output_projection_cuda_aot_candidate(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  request_shape~ : @joint_diffusion_plan.JointDiffusionRequestShape,
  requirement_ordinal~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : JointDiffusionCudaAotLimits,
) -> JointDiffusionOutputProjectionCudaCandidate raise JointDiffusionCudaAotError {
  if !target.supports_bf16() {
    raise Invalid(Target)
  }
  if compiler.reassociate() || compiler.fused_multiply_add() {
    raise Invalid(Compiler)
  }
  validate_request_shape(requirements, request_shape)
  let projection = requirements.projection_contract()
  if projection.output_storage() != Float32InputWeightBiasFloat32Output ||
    projection.video_layout() != BatchFrameHeightWidthChannelTemporalHeightWidth ||
    projection.audio_layout() != ChannelMajorTime {
    raise Invalid(Requirement)
  }
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let shapes = requirements.shape_contract()
  let (latent, rows, input_width, output_width, input_role, output_role, symbol) : (
    @joint_diffusion_plan.JointDiffusionLatentKind,
    Int64,
    Int,
    Int,
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperandRole,
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperandRole,
    String,
  ) = match (requirement.operation(), requirement.capability()) {
    (
      VideoOutputProjection(channels, patch_t, patch_h, patch_w, hidden),
      VideoOutputProjectionAot,
    ) => {
      if channels != shapes.video_latent_channels() ||
        patch_t != shapes.video_patch_temporal() ||
        patch_h != shapes.video_patch_height() ||
        patch_w != shapes.video_patch_width() ||
        hidden != requirements.denoiser_shape().hidden_size() ||
        request_shape.video_latent_frames() % patch_t != 0 ||
        request_shape.video_latent_height() % patch_h != 0 ||
        request_shape.video_latent_width() % patch_w != 0 {
        raise Invalid(Requirement)
      }
      let row_count = checked_mul(
        checked_mul(
          (request_shape.video_latent_frames() / patch_t).to_int64(),
          (request_shape.video_latent_height() / patch_h).to_int64(),
        ),
        (request_shape.video_latent_width() / patch_w).to_int64(),
      )
      let width = checked_mul(
        checked_mul(channels.to_int64(), patch_t.to_int64()),
        checked_mul(patch_h.to_int64(), patch_w.to_int64()),
      )
      if width > 2147483647L {
        raise SizeOverflow
      }
      (
        VideoLatent,
        row_count,
        hidden,
        width.to_int(),
        VideoF32HeadInput,
        VideoPredictionOutput,
        "lunaflux_joint_diffusion_video_output_projection_f32_v1",
      )
    }
    (AudioOutputProjection(channels, hidden), AudioOutputProjectionAot) => {
      if channels != shapes.audio_latent_channels() ||
        hidden != requirements.denoiser_shape().hidden_size() {
        raise Invalid(Requirement)
      }
      (
        AudioLatent,
        request_shape.audio_token_rows().to_int64(),
        hidden,
        channels,
        AudioF32HeadInput,
        AudioPredictionOutput,
        "lunaflux_joint_diffusion_audio_output_projection_f32_v1",
      )
    }
    _ => raise UnsupportedRequirement(requirement_ordinal)
  }
  if rows <= 0L || input_width <= 0 || output_width <= 0 {
    raise Invalid(Requirement)
  }
  let input_elements = checked_mul(rows, input_width.to_int64())
  let output_elements = checked_mul(rows, output_width.to_int64())
  let weight_elements = checked_mul(
    input_width.to_int64(),
    output_width.to_int64(),
  )
  let bias_elements = output_width.to_int64()
  if input_elements > limits.maximum_elements ||
    output_elements > limits.maximum_elements ||
    weight_elements > limits.maximum_elements ||
    bias_elements > limits.maximum_elements {
    raise Invalid(Limit)
  }
  let grid = (output_elements + 255L) / 256L
  if grid > 2147483647L {
    raise SizeOverflow
  }
  let dimensions = @launch_contract.AotLaunchDimensions::new(
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
  let operands = [
    launch_operand(input_role, checked_mul(input_elements, 4L), 16L),
    launch_operand(ProjectionWeight, checked_mul(weight_elements, 4L), 16L),
    launch_operand(ProjectionBias, checked_mul(bias_elements, 4L), 16L),
    launch_operand(output_role, checked_mul(output_elements, 4L), 16L),
  ]
  let rendered = @decoder_projection_cuda_source.dense_projection_f32_bias_f32_source(
    rows~,
    input_width~,
    output_width~,
    function_symbol=symbol,
  ) catch {
    error => raise Source(error)
  }
  if rendered.operation() != DenseProjectionF32BiasF32 ||
    rendered.numerical_contract() != OrderedFloat32AccumulationF32OutputV1 ||
    rendered.row_count() != rows ||
    rendered.input_width() != input_width ||
    rendered.output_width() != output_width ||
    rendered.function_symbol() != symbol {
    abort("shared F32 output projection renderer violated its typed contract")
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = JointDiffusionCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let encoded_recipe = output_projection_recipe(
    requirements, requirement_ordinal, latent, request_shape, rows, input_width,
    output_width, input_elements, output_elements, weight_elements, bias_elements,
    target, compiler, dimensions, operands, source_digest,
  )
  if encoded_recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    numerical_contract: OrderedFloat32DenseF32OutputV1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    latent,
    request_shape,
    row_count: rows,
    input_width,
    output_width,
    input_element_count: input_elements,
    output_element_count: output_elements,
    projection_weight_element_count: weight_elements,
    projection_bias_element_count: bias_elements,
    target,
    dimensions,
    operands: ReadOnlyArray::from_array(operands),
    compiler,
    function_symbol: symbol,
    source,
    source_digest,
    recipe: encoded_recipe,
    recipe_digest: JointDiffusionCudaRecipeDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(encoded_recipe)),
    ),
  }
}
