///|
pub fn lower_timestep_mlp_cuda_aot_candidate(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  values~ : @joint_diffusion_io_plan.JointDistinctTimestepValues,
  requirement_ordinal~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : JointDiffusionCudaAotLimits,
) -> JointDiffusionTimestepMlpCudaCandidate raise JointDiffusionCudaAotError {
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
  let (input, hidden, output) = match
    (requirement.operation(), requirement.capability()) {
    (TimestepEmbeddingMlpRows(i, h, o), TimestepEmbeddingMlpRowsAot) =>
      (i, h, o)
    _ => raise UnsupportedRequirement(requirement_ordinal)
  }
  let denoiser = requirements.denoiser_shape()
  if input != denoiser.timestep_frequency_dimension() ||
    hidden != denoiser.timestep_embedding_hidden_dimension() ||
    output != denoiser.final_adaln_conditioning_dimension() {
    raise Invalid(Requirement)
  }
  let rows = values.values().length()
  let w1 = checked_mul(hidden.to_int64(), input.to_int64())
  let w2 = checked_mul(output.to_int64(), hidden.to_int64())
  let maximum = if w1 > w2 { w1 } else { w2 }
  if maximum > limits.maximum_elements {
    raise Invalid(Limit)
  }
  let elements = checked_mul(rows.to_int64(), output.to_int64())
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
    launch_operand(
      TimestepFrequencyRowsOutput,
      rows.to_int64() * input.to_int64() * 4L,
      16L,
    ),
    launch_operand(TimestepMlpWeight1, w1 * 4L, 16L),
    launch_operand(TimestepMlpBias1, hidden.to_int64() * 4L, 16L),
    launch_operand(TimestepMlpWeight2, w2 * 4L, 16L),
    launch_operand(TimestepMlpBias2, output.to_int64() * 4L, 16L),
    launch_operand(TimestepEmbeddingRowsOutput, elements * 4L, 16L),
  ]
  let symbol = "lunaflux_joint_diffusion_timestep_mlp_f32_v1"
  let rendered = @timestep_embedding_cuda_source.dense_silu_dense_f32_rows_source(
    rows~,
    input_width=input,
    hidden_width=hidden,
    output_width=output,
    function_symbol=symbol,
  ) catch {
    _ => raise Invalid(Encoding)
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
    "lunaflux.joint-timestep-mlp.v1|content=" +
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
    ":" +
    output.to_string() +
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
    ":256:0|abi=" +
    abi +
    "|source=" +
    source_digest.as_hex(),
  )
  if source.length() > limits.maximum_source_bytes ||
    recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    numerical_contract: OrderedFloat32DenseSiluDenseV1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    values_digest: values.digest(),
    row_count: rows,
    input_width: input,
    hidden_width: hidden,
    output_width: output,
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
