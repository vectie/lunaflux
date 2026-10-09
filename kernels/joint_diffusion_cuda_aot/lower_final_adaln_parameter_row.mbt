///|
pub fn lower_final_adaln_parameter_row_cuda_aot_candidate(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  requirement_ordinal~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : JointDiffusionCudaAotLimits,
) -> JointDiffusionFinalAdaLnParameterRowCudaCandidate raise JointDiffusionCudaAotError {
  if !target.supports_bf16() {
    raise Invalid(Target)
  }
  if compiler.reassociate() || compiler.fused_multiply_add() {
    raise Invalid(Compiler)
  }
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let (input_width, hidden) = match
    (requirement.operation(), requirement.capability()) {
    (FinalAdaLnParameterRow(input, output), FinalAdaLnParameterRowAot) =>
      (input, output)
    _ => raise UnsupportedRequirement(requirement_ordinal)
  }
  let denoiser = requirements.denoiser_shape()
  if input_width != denoiser.final_adaln_conditioning_dimension() ||
    hidden != denoiser.hidden_size() {
    raise Invalid(Requirement)
  }
  let outputs = checked_mul(hidden.to_int64(), 2L)
  let weights = checked_mul(outputs, input_width.to_int64())
  if weights > limits.maximum_elements {
    raise Invalid(Limit)
  }
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=outputs.to_int(),
    grid_y=1,
    grid_z=1,
    block_x=1,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    error => raise Launch(error)
  }
  let operands = [
    launch_operand(
      FinalAdaLnConditioningInput,
      checked_mul(input_width.to_int64(), 4L),
      16L,
    ),
    launch_operand(FinalAdaLnWeight, checked_mul(weights, 2L), 16L),
    launch_operand(FinalAdaLnBias, checked_mul(outputs, 2L), 16L),
    launch_operand(
      FinalAdaLnShiftOutput,
      checked_mul(hidden.to_int64(), 2L),
      16L,
    ),
    launch_operand(
      FinalAdaLnScaleOutput,
      checked_mul(hidden.to_int64(), 2L),
      16L,
    ),
  ]
  let symbol = "lunaflux_joint_diffusion_final_adaln_parameter_row_bf16_v1"
  let rendered = @affine_modulation_cuda_source.final_adaln_parameter_row_source(
    input_width~,
    hidden_size=hidden,
    function_symbol=symbol,
  ) catch {
    error => raise AffineSource(error)
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = JointDiffusionCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let compiler_id = compiler.identity()
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
  let recipe = @utf8.encode(
    "lunaflux.joint-diffusion-final-adaln-parameter-row.v1|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|ordinal=" +
    requirement_ordinal.to_string() +
    "|geometry=" +
    input_width.to_string() +
    ":" +
    hidden.to_string() +
    ":" +
    weights.to_string() +
    "|sm=" +
    target.compute_major().to_string() +
    "." +
    target.compute_minor().to_string() +
    ":" +
    target.supports_bf16().to_string() +
    ":" +
    target.supports_cublas_lt().to_string() +
    "|compiler=" +
    compiler_id.toolchain_digest_sha256() +
    ":" +
    compiler_id.major().to_string() +
    "." +
    compiler_id.minor().to_string() +
    "." +
    compiler_id.patch().to_string() +
    "|opt=" +
    compiler.optimization_level().to_string() +
    "|fma=" +
    compiler.fused_multiply_add().to_string() +
    "|reassociate=" +
    compiler.reassociate().to_string() +
    "|max-registers=" +
    compiler.max_registers().to_string() +
    "|launch=" +
    outputs.to_string() +
    ":1:0|operands=" +
    operand_text.to_string() +
    "|source=" +
    source_digest.as_hex(),
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    numerical_contract: SiluF32CastBf16OrderedDenseBf16OutputV1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    input_width,
    hidden_size: hidden,
    weight_element_count: weights,
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
