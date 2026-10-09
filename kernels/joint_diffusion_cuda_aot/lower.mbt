///|
fn recipe(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  ordinal : Int,
  latent : @joint_diffusion_plan.JointDiffusionLatentKind,
  shift_numerator : Int,
  shift_denominator : Int,
  training_timestep_count : Int,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
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
    "lunaflux.joint-diffusion-flow-cuda-aot.v2" +
    "|numerical=ordered-f32-rf-velocity-eta0-v1" +
    "|storage=f32-state,f32-velocity,f32-sigma-t-ratio-complement,f32-output" +
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
    "|scheduler=" +
    shift_numerator.to_string() +
    "/" +
    shift_denominator.to_string() +
    ":" +
    training_timestep_count.to_string() +
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
    dimensions.grid_y().to_string() +
    ":" +
    dimensions.grid_z().to_string() +
    ":" +
    dimensions.block_x().to_string() +
    ":" +
    dimensions.block_y().to_string() +
    ":" +
    dimensions.block_z().to_string() +
    ":" +
    dimensions.shared_memory_bytes().to_string() +
    "|operands=" +
    operand_text.to_string() +
    "|source=" +
    source_digest.as_hex(),
  )
}

///|
/// Lowers one exact video or audio shifted-flow Euler update into deterministic
/// F32 CUDA source. The request shape is authenticated against the plan-owned
/// shape contract and becomes part of the canonical recipe identity.
pub fn lower_shifted_flow_cuda_aot_candidate(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  request_shape~ : @joint_diffusion_plan.JointDiffusionRequestShape,
  requirement_ordinal~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : JointDiffusionCudaAotLimits,
) -> JointDiffusionShiftedFlowCudaCandidate raise JointDiffusionCudaAotError {
  if compiler.reassociate() || compiler.fused_multiply_add() {
    raise Invalid(Compiler)
  }
  validate_request_shape(requirements, request_shape)
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let (latent, shift_numerator, shift_denominator, training_timestep_count) = match
    (requirement.operation(), requirement.capability()) {
    (
      ShiftedFlowStep(latent, numerator, denominator, timesteps),
      ShiftedFlowStepAot,
    ) => (latent, numerator, denominator, timesteps)
    _ => raise UnsupportedRequirement(requirement_ordinal)
  }
  if shift_numerator <= 0 ||
    shift_denominator <= 0 ||
    training_timestep_count <= 0 {
    raise Invalid(Requirement)
  }
  let elements = element_count(requirements, request_shape, latent)
  if elements <= 0L || elements > limits.maximum_elements {
    raise Invalid(Limit)
  }
  let grid = (elements + 255L) / 256L
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
  let latent_bytes = checked_mul(elements, 4L)
  let (input_role, prediction_role, output_role, symbol) : (
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperandRole,
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperandRole,
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperandRole,
    String,
  ) = match latent {
    VideoLatent =>
      (
        VideoLatentInput,
        VideoPredictionInput,
        VideoLatentOutput,
        "lunaflux_rectified_flow_f32_eta0_v1",
      )
    AudioLatent =>
      (
        AudioLatentInput,
        AudioPredictionInput,
        AudioLatentOutput,
        "lunaflux_rectified_flow_f32_eta0_v1",
      )
  }
  let operands = [
    launch_operand(input_role, latent_bytes, 16L),
    launch_operand(prediction_role, latent_bytes, 16L),
    launch_operand(FlowCoefficientsInput, 12L, 4L),
    launch_operand(output_role, latent_bytes, 16L),
  ]
  let source = @rectified_flow_update_cuda_source.render(
    elements~,
    block_size=256,
  ) catch {
    _ => raise Invalid(Limit)
  }
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = JointDiffusionCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let encoded_recipe = recipe(
    requirements, requirement_ordinal, latent, shift_numerator, shift_denominator,
    training_timestep_count, request_shape, elements, target, compiler, dimensions,
    operands, source_digest,
  )
  if encoded_recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    numerical_contract: OrderedFloat32RfVelocityEtaZeroV1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    latent,
    request_shape,
    element_count: elements,
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
