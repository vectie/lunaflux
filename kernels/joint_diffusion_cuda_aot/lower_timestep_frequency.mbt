///|
pub fn lower_timestep_frequency_cuda_aot_candidate(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  values~ : @joint_diffusion_io_plan.JointDistinctTimestepValues,
  requirement_ordinal~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : JointDiffusionCudaAotLimits,
) -> JointDiffusionTimestepFrequencyCudaCandidate raise JointDiffusionCudaAotError {
  if values.identity() != requirements.source_identity() {
    raise Invalid(RequestShape)
  }
  if compiler.reassociate() || compiler.fused_multiply_add() {
    raise Invalid(Compiler)
  }
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let width = match (requirement.operation(), requirement.capability()) {
    (TimestepFrequencyRows(value), TimestepFrequencyRowsAot) => value
    _ => raise UnsupportedRequirement(requirement_ordinal)
  }
  if width != requirements.denoiser_shape().timestep_frequency_dimension() {
    raise Invalid(Requirement)
  }
  let rows = values.values().length()
  let elements = checked_mul(rows.to_int64(), width.to_int64())
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
    launch_operand(DistinctTimestepValuesInput, rows.to_int64() * 4L, 16L),
    launch_operand(TimestepFrequencyRowsOutput, elements * 4L, 16L),
  ]
  let symbol = "lunaflux_joint_diffusion_timestep_frequency_rows_f32_v1"
  let rendered = @timestep_embedding_cuda_source.cosine_sine_timestep_rows_source(
    rows~,
    frequency_width=width,
    function_symbol=symbol,
  ) catch {
    _ => raise Invalid(Encoding)
  }
  let source = rendered.source_bytes()
  let source_digest = JointDiffusionCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let recipe = @utf8.encode(
    "lunaflux.joint-timestep-frequency.v1|content=" +
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
    width.to_string() +
    "|sm=" +
    target.compute_major().to_string() +
    "." +
    target.compute_minor().to_string() +
    "|compiler=" +
    compiler.identity().toolchain_digest_sha256() +
    "|launch=" +
    dimensions.grid_x().to_string() +
    ":256:0|source=" +
    source_digest.as_hex(),
  )
  if source.length() > limits.maximum_source_bytes ||
    recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    numerical_contract: Float32CosSinExpfPeriod10000V1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    values_digest: values.digest(),
    row_count: rows,
    frequency_width: width,
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
