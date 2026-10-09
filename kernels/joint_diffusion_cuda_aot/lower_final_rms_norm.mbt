///|
fn final_rms_norm_recipe(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  ordinal : Int,
  latent : @joint_diffusion_plan.JointDiffusionLatentKind,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
  rows : Int64,
  hidden : Int,
  numerator : Int,
  denominator : Int,
  elements : Int64,
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
    "|numerical=ordered-f32-rms-norm-bf16-round-v1" +
    "|storage=selected-bf16-hidden,bf16-final-norm-weight,bf16-normalized-hidden" +
    "|scope=rms-norm-only-before-adaln-modulation" +
    "|selection=target-modality-before-row-independent-rms-norm" +
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
    "|rms=" +
    rows.to_string() +
    ":" +
    hidden.to_string() +
    ":" +
    numerator.to_string() +
    "/" +
    denominator.to_string() +
    ":" +
    elements.to_string() +
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
    "|source=" +
    source_digest.as_hex(),
  )
}

///|
/// Lowers only the final row-wise BF16 RMSNorm for selected target video or
/// audio rows. Row selection commutes with RMSNorm because each row is reduced
/// independently with the same authenticated weight and epsilon.
pub fn lower_final_rms_norm_cuda_aot_candidate(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  request_shape~ : @joint_diffusion_plan.JointDiffusionRequestShape,
  requirement_ordinal~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : JointDiffusionCudaAotLimits,
) -> JointDiffusionFinalRmsNormCudaCandidate raise JointDiffusionCudaAotError {
  if !target.supports_bf16() {
    raise Invalid(Target)
  }
  if compiler.reassociate() || compiler.fused_multiply_add() {
    raise Invalid(Compiler)
  }
  validate_request_shape(requirements, request_shape)
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let shape = requirements.shape_contract()
  let denoiser = requirements.denoiser_shape()
  let (
    latent,
    rows,
    hidden,
    numerator,
    denominator,
    input_role,
    output_role,
    symbol,
  ) : (
    @joint_diffusion_plan.JointDiffusionLatentKind,
    Int64,
    Int,
    Int,
    Int,
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperandRole,
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperandRole,
    String,
  ) = match (requirement.operation(), requirement.capability()) {
    (
      VideoFinalRmsNorm(
        operation_hidden,
        operation_numerator,
        operation_denominator
      ),
      VideoFinalRmsNormAot,
    ) => {
      let patch_t = shape.video_patch_temporal()
      let patch_h = shape.video_patch_height()
      let patch_w = shape.video_patch_width()
      if operation_hidden != denoiser.hidden_size() ||
        operation_numerator != denoiser.final_norm_epsilon_numerator() ||
        operation_denominator != denoiser.final_norm_epsilon_denominator() ||
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
      (
        VideoLatent,
        row_count,
        operation_hidden,
        operation_numerator,
        operation_denominator,
        VideoFinalNormInput,
        VideoFinalNormOutput,
        "lunaflux_joint_diffusion_video_final_rms_norm_bf16_v1",
      )
    }
    (
      AudioFinalRmsNorm(
        operation_hidden,
        operation_numerator,
        operation_denominator
      ),
      AudioFinalRmsNormAot,
    ) => {
      if operation_hidden != denoiser.hidden_size() ||
        operation_numerator != denoiser.final_norm_epsilon_numerator() ||
        operation_denominator != denoiser.final_norm_epsilon_denominator() {
        raise Invalid(Requirement)
      }
      (
        AudioLatent,
        request_shape.audio_token_rows().to_int64(),
        operation_hidden,
        operation_numerator,
        operation_denominator,
        AudioFinalNormInput,
        AudioFinalNormOutput,
        "lunaflux_joint_diffusion_audio_final_rms_norm_bf16_v1",
      )
    }
    _ => raise UnsupportedRequirement(requirement_ordinal)
  }
  let elements = checked_mul(rows, hidden.to_int64())
  if elements > limits.maximum_elements ||
    hidden.to_int64() > limits.maximum_elements {
    raise Invalid(Limit)
  }
  if rows > 2147483647L {
    raise SizeOverflow
  }
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=rows.to_int(),
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=1024,
  ) catch {
    error => raise Launch(error)
  }
  let operands = [
    launch_operand(input_role, checked_mul(elements, 2L), 16L),
    launch_operand(FinalNormWeight, checked_mul(hidden.to_int64(), 2L), 16L),
    launch_operand(output_role, checked_mul(elements, 2L), 16L),
  ]
  let epsilon = numerator.to_double() / denominator.to_double()
  let rendered = @decoder_foundation_cuda_source.rms_norm_bf16_fixed_rows_source(
    rows~,
    hidden_size=hidden,
    epsilon~,
    function_symbol=symbol,
  ) catch {
    error => raise FoundationSource(error)
  }
  if rendered.operation() != RmsNormBf16 ||
    rendered.numerical_contract() != OrderedFloat32ReductionBf16RoundV1 ||
    rendered.function_symbol() != symbol {
    abort("shared fixed-row RMSNorm renderer violated its typed contract")
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = JointDiffusionCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let encoded_recipe = final_rms_norm_recipe(
    requirements, requirement_ordinal, latent, request_shape, rows, hidden, numerator,
    denominator, elements, target, compiler, dimensions, operands, source_digest,
  )
  if encoded_recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    numerical_contract: OrderedFloat32RmsNormBf16RoundV1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    latent,
    request_shape,
    row_count: rows,
    hidden_size: hidden,
    epsilon_numerator: numerator,
    epsilon_denominator: denominator,
    element_count: elements,
    weight_element_count: hidden.to_int64(),
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
