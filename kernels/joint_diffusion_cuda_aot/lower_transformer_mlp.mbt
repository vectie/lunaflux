///|
fn transformer_mlp_operands(
  rows : Int,
  hidden : Int,
  packed_weights : Int64,
  down_weights : Int64,
) -> Array[@joint_diffusion_artifact_admission.JointDiffusionLaunchOperand] raise JointDiffusionCudaAotError {
  let elements = checked_mul(rows.to_int64(), hidden.to_int64())
  [
    launch_operand(JointHiddenInput, checked_mul(elements, 2L), 16L),
    launch_operand(
      TransformerMlpPackedGateUpWeight,
      checked_mul(packed_weights, 2L),
      16L,
    ),
    launch_operand(TransformerMlpDownWeight, checked_mul(down_weights, 2L), 16L),
    launch_operand(JointHiddenOutput, checked_mul(elements, 2L), 16L),
  ]
}

///|
fn transformer_mlp_dimensions(
  rows : Int,
  hidden : Int,
) -> @launch_contract.AotLaunchDimensions raise JointDiffusionCudaAotError {
  let elements = checked_mul(rows.to_int64(), hidden.to_int64())
  let grid = (elements + 255L) / 256L
  if grid > 2147483647L {
    raise Invalid(Limit)
  }
  @launch_contract.AotLaunchDimensions::new(
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
}

///|
fn transformer_mlp_operand_text(
  operands : ArrayView[
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
  ],
) -> String {
  let writer = StringBuilder()
  for index, operand in operands {
    if index > 0 {
      writer.write_char(',')
    }
    writer.write_string(role_tag(operand.role()))
    writer.write_char('@')
    writer.write_string(operand.byte_count().to_string())
    writer.write_char('/')
    writer.write_string(operand.alignment().to_string())
  }
  writer.to_string()
}

///|
pub fn lower_transformer_mlp_cuda_aot_candidate(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  request~ : @joint_diffusion_io_plan.JointDiffusionIoPlan,
  packed_timestep~ : @joint_diffusion_io_plan.JointPackedTimestepSelection,
  requirement_ordinal~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : JointDiffusionCudaAotLimits,
) -> JointDiffusionTransformerMlpCudaCandidate raise JointDiffusionCudaAotError {
  if request.identity() != requirements.source_identity() ||
    packed_timestep.identity() != request.identity() ||
    packed_timestep.request_digest() != request.request_digest() {
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
  let (layers, refiners, hidden, intermediate) = match
    (requirement.operation(), requirement.capability()) {
    (JointTransformerMlp(l, r, h, i), JointTransformerMlpAot) => (l, r, h, i)
    _ => raise UnsupportedRequirement(requirement_ordinal)
  }
  let denoiser = requirements.denoiser_shape()
  if layers != denoiser.layer_count() ||
    refiners != denoiser.refiner_layer_count() ||
    hidden != denoiser.hidden_size() ||
    intermediate != denoiser.feed_forward_dimension() {
    raise Invalid(Requirement)
  }
  let packed_rows = packed_timestep.packed_row_count()
  let refiner_rows = request.text().token_count()
  let packed_weight_elements = checked_mul(
    checked_mul(intermediate.to_int64(), 2L),
    hidden.to_int64(),
  )
  let down_weight_elements = checked_mul(
    hidden.to_int64(),
    intermediate.to_int64(),
  )
  let packed_elements = checked_mul(packed_rows.to_int64(), hidden.to_int64())
  let refiner_elements = checked_mul(refiner_rows.to_int64(), hidden.to_int64())
  if packed_weight_elements > limits.maximum_elements ||
    down_weight_elements > limits.maximum_elements ||
    packed_elements > limits.maximum_elements ||
    refiner_elements > limits.maximum_elements {
    raise Invalid(Limit)
  }
  let packed_dimensions = transformer_mlp_dimensions(packed_rows, hidden)
  let refiner_dimensions = transformer_mlp_dimensions(refiner_rows, hidden)
  let packed_operands = transformer_mlp_operands(
    packed_rows, hidden, packed_weight_elements, down_weight_elements,
  )
  let refiner_operands = transformer_mlp_operands(
    refiner_rows, hidden, packed_weight_elements, down_weight_elements,
  )
  let packed_rendered = @gated_mlp_cuda_source.packed_swiglu_dense_bf16_rows_source(
    rows=packed_rows,
    hidden_size=hidden,
    intermediate_size=intermediate,
    function_symbol="lunaflux_joint_transformer_mlp_packed_bf16_v1",
  ) catch {
    error => raise GatedMlpSource(error)
  }
  let refiner_rendered = @gated_mlp_cuda_source.packed_swiglu_dense_bf16_rows_source(
    rows=refiner_rows,
    hidden_size=hidden,
    intermediate_size=intermediate,
    function_symbol="lunaflux_joint_transformer_mlp_refiner_bf16_v1",
  ) catch {
    error => raise GatedMlpSource(error)
  }
  let source = packed_rendered.source_bytes() + refiner_rendered.source_bytes()
  let source_digest = JointDiffusionCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let compiler_id = compiler.identity()
  let packed_abi = transformer_mlp_operand_text(packed_operands)
  let refiner_abi = transformer_mlp_operand_text(refiner_operands)
  let recipe = @utf8.encode(
    "lunaflux.joint-transformer-mlp.v1|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|request=" +
    request.request_digest().as_hex() +
    "|packed-timestep=" +
    packed_timestep.digest().as_hex() +
    "|ordinal=" +
    requirement_ordinal.to_string() +
    "|layers=" +
    layers.to_string() +
    ":" +
    refiners.to_string() +
    "|geometry=" +
    packed_rows.to_string() +
    ":" +
    refiner_rows.to_string() +
    ":" +
    hidden.to_string() +
    ":" +
    intermediate.to_string() +
    "|weights=packed-up-gate-bf16[" +
    (intermediate * 2).to_string() +
    "," +
    hidden.to_string() +
    "]:down-bf16[" +
    hidden.to_string() +
    "," +
    intermediate.to_string() +
    "]" +
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
    packed_dimensions.grid_x().to_string() +
    ":" +
    refiner_dimensions.grid_x().to_string() +
    ":256:0|packed-abi=" +
    packed_abi +
    "|refiner-abi=" +
    refiner_abi +
    "|source=" +
    source_digest.as_hex(),
  )
  if source.length() > limits.maximum_source_bytes ||
    recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    numerical_contract: OrderedF32AccumulationBf16SwiGluDownBf16V1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    request_digest: request.request_digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    layer_count: layers,
    refiner_layer_count: refiners,
    packed_row_count: packed_rows,
    refiner_row_count: refiner_rows,
    hidden_size: hidden,
    intermediate_size: intermediate,
    packed_weight_element_count: packed_weight_elements,
    down_weight_element_count: down_weight_elements,
    packed_dimensions,
    refiner_dimensions,
    packed_operands: ReadOnlyArray::from_array(packed_operands),
    refiner_operands: ReadOnlyArray::from_array(refiner_operands),
    source,
    source_digest,
    recipe,
    recipe_digest: JointDiffusionCudaRecipeDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(recipe)),
    ),
  }
}
