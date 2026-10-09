///|
pub fn lower_distinct_timestep_adaln_cuda_aot_candidate(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  values~ : @joint_diffusion_io_plan.JointDistinctTimestepValues,
  requirement_ordinal~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : JointDiffusionCudaAotLimits,
) -> JointDiffusionDistinctTimestepAdaLnCudaCandidate raise JointDiffusionCudaAotError {
  if values.identity() != requirements.source_identity() {
    raise Invalid(RequestShape)
  }
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
  let (input, hidden) = match
    (requirement.operation(), requirement.capability()) {
    (
      DistinctTimestepAdaLnParameterRows(i, h),
      DistinctTimestepAdaLnParameterRowsAot,
    ) => (i, h)
    _ => raise UnsupportedRequirement(requirement_ordinal)
  }
  let denoiser = requirements.denoiser_shape()
  if input != denoiser.final_adaln_conditioning_dimension() ||
    hidden != denoiser.hidden_size() {
    raise Invalid(Requirement)
  }
  let rows = values.values().length()
  let outputs_per_row = checked_mul(hidden.to_int64(), 2L)
  let output_elements = checked_mul(rows.to_int64(), outputs_per_row)
  let weights = checked_mul(outputs_per_row, input.to_int64())
  if weights > limits.maximum_elements ||
    output_elements > limits.maximum_elements ||
    output_elements > 2147483647L {
    raise Invalid(Limit)
  }
  let input_elements = checked_mul(rows.to_int64(), input.to_int64())
  let table_elements = checked_mul(rows.to_int64(), hidden.to_int64())
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=output_elements.to_int(),
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
      TimestepEmbeddingRowsOutput,
      checked_mul(input_elements, 4L),
      16L,
    ),
    launch_operand(FinalAdaLnWeight, checked_mul(weights, 2L), 16L),
    launch_operand(FinalAdaLnBias, checked_mul(outputs_per_row, 2L), 16L),
    launch_operand(FinalAdaLnShiftOutput, checked_mul(table_elements, 2L), 16L),
    launch_operand(FinalAdaLnScaleOutput, checked_mul(table_elements, 2L), 16L),
  ]
  let symbol = "lunaflux_joint_diffusion_distinct_timestep_adaln_bf16_v1"
  let rendered = @affine_modulation_cuda_source.final_adaln_parameter_rows_source(
    rows~,
    input_width=input,
    hidden_size=hidden,
    function_symbol=symbol,
  ) catch {
    error => raise AffineSource(error)
  }
  let source = rendered.source_bytes()
  let source_digest = JointDiffusionCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let compiler_id = compiler.identity()
  let abi = operands
    .map(operand => {
      role_tag(operand.role()) +
      ":" +
      operand.byte_count().to_string() +
      ":" +
      operand.alignment().to_string()
    })
    .join(",")
  let recipe = @utf8.encode(
    "lunaflux.joint-distinct-timestep-adaln.v1|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|ordinal=" +
    requirement_ordinal.to_string() +
    "|values=" +
    values.digest().as_hex() +
    "|geometry=" +
    rows.to_string() +
    ":" +
    input.to_string() +
    ":" +
    hidden.to_string() +
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
    dimensions.grid_x().to_string() +
    ":1:0|abi=" +
    abi +
    "|source=" +
    source_digest.as_hex(),
  )
  if source.length() > limits.maximum_source_bytes ||
    recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    numerical_contract: SiluF32CastBf16OrderedDenseBf16OutputV1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    values_digest: values.digest(),
    row_count: rows,
    input_width: input,
    hidden_size: hidden,
    weight_element_count: weights,
    dimensions,
    operands: ReadOnlyArray::from_array(operands),
    source,
    source_digest,
    recipe,
    recipe_digest: JointDiffusionCudaRecipeDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(recipe)),
    ),
  }
}
