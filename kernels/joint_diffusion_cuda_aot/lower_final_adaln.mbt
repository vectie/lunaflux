///|
pub fn lower_final_adaln_f32_cuda_aot_candidate(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  request_shape~ : @joint_diffusion_plan.JointDiffusionRequestShape,
  requirement_ordinal~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : JointDiffusionCudaAotLimits,
) -> JointDiffusionFinalAdaLnF32CudaCandidate raise JointDiffusionCudaAotError {
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
  let (latent, rows, hidden, normalized_role, output_role, symbol) : (
    @joint_diffusion_plan.JointDiffusionLatentKind,
    Int64,
    Int,
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperandRole,
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperandRole,
    String,
  ) = match (requirement.operation(), requirement.capability()) {
    (VideoFinalAdaLnF32Boundary(operation_hidden), VideoFinalAdaLnF32BoundaryAot
    ) => {
      if operation_hidden != denoiser.hidden_size() ||
        request_shape.video_latent_frames() % shape.video_patch_temporal() != 0 ||
        request_shape.video_latent_height() % shape.video_patch_height() != 0 ||
        request_shape.video_latent_width() % shape.video_patch_width() != 0 {
        raise Invalid(Requirement)
      }
      let rows = checked_mul(
        checked_mul(
          (request_shape.video_latent_frames() / shape.video_patch_temporal()).to_int64(),
          (request_shape.video_latent_height() / shape.video_patch_height()).to_int64(),
        ),
        (request_shape.video_latent_width() / shape.video_patch_width()).to_int64(),
      )
      (
        VideoLatent,
        rows,
        operation_hidden,
        VideoFinalNormOutput,
        VideoF32HeadInput,
        "lunaflux_joint_diffusion_video_final_adaln_f32_v1",
      )
    }
    (AudioFinalAdaLnF32Boundary(operation_hidden), AudioFinalAdaLnF32BoundaryAot
    ) => {
      if operation_hidden != denoiser.hidden_size() {
        raise Invalid(Requirement)
      }
      (
        AudioLatent,
        request_shape.audio_token_rows().to_int64(),
        operation_hidden,
        AudioFinalNormOutput,
        AudioF32HeadInput,
        "lunaflux_joint_diffusion_audio_final_adaln_f32_v1",
      )
    }
    _ => raise UnsupportedRequirement(requirement_ordinal)
  }
  let elements = checked_mul(rows, hidden.to_int64())
  if elements > limits.maximum_elements {
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
  let operands = [
    launch_operand(normalized_role, checked_mul(elements, 2L), 16L),
    launch_operand(FinalAdaLnShiftInput, checked_mul(elements, 2L), 16L),
    launch_operand(FinalAdaLnScaleInput, checked_mul(elements, 2L), 16L),
    launch_operand(output_role, checked_mul(elements, 4L), 16L),
  ]
  let rendered = @affine_modulation_cuda_source.bf16_shift_scale_f32_fixed_rows_source(
    rows~,
    width=hidden,
    function_symbol=symbol,
  ) catch {
    error => raise AffineSource(error)
  }
  if rendered.operation() != Bf16ShiftScaleToF32 ||
    rendered.numerical_contract() != StagedBf16ShiftScaleF32OutputV1 {
    abort("shared affine renderer violated its typed contract")
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = JointDiffusionCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let latent_tag = match latent {
    VideoLatent => "video"
    AudioLatent => "audio"
  }
  let operand_text = StringBuilder()
  for index, operand in operands {
    if index > 0 {
      operand_text.write_char(',')
    }
    operand_text.write_string(role_tag(operand.role()))
    operand_text.write_char('@')
    operand_text.write_string(operand.byte_count().to_string())
    operand_text.write_char('/')
    operand_text.write_string(operand.alignment().to_string())
  }
  let compiler_identity = compiler.identity()
  let recipe = @utf8.encode(
    "lunaflux.joint-diffusion-final-adaln-f32.v1|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|ordinal=" +
    requirement_ordinal.to_string() +
    "|latent=" +
    latent_tag +
    "|rows=" +
    rows.to_string() +
    "|hidden=" +
    hidden.to_string() +
    "|elements=" +
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
    "|symbol=" +
    symbol +
    "|source=" +
    source_digest.as_hex(),
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    numerical_contract: StagedBf16AdaLnF32BoundaryV1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    latent,
    request_shape,
    row_count: rows,
    hidden_size: hidden,
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
