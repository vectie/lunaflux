///|
fn compressed_index_operand_text(
  operands : ArrayView[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ],
) -> String {
  let text = StringBuilder()
  for index, item in operands {
    if index > 0 {
      text.write_char(',')
    }
    let role = match item.role() {
      StepCountsInput => "step-counts"
      CompressedQueryPositionsInput => "compressed-query-positions"
      CompressedCacheOffsetsInput => "compressed-cache-offsets"
      CompressedIndexCountsOutput => "compressed-index-counts"
      CompressedIndicesOutput => "compressed-indices"
      _ => abort("unexpected compressed-index operand role")
    }
    text.write_string(role)
    text.write_char('@')
    text.write_string(item.byte_count().to_string())
    text.write_char('/')
    text.write_string(item.alignment().to_string())
  }
  text.to_string()
}

///|
fn compressed_index_recipe(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  ordinal : Int,
  maximum_tokens : Int,
  compression_ratio : Int,
  maximum_position_count : Int,
  maximum_slots_per_row : Int,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  dimensions : @launch_contract.AotLaunchDimensions,
  operands : ArrayView[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ],
  source_digest : AdvancedDecoderCudaSourceDigest,
) -> Bytes {
  let compiler_identity = compiler.identity()
  @utf8.encode(
    "lunaflux.advanced-decoder-compressed-index-cuda-aot.v1" +
    "|family=causal-complete-slots-i32" +
    "|numerical=bit-exact-causal-complete-slots-i32-v1" +
    "|layout=dense-row-major-prefix-minus-one-padding" +
    "|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|ordinal=" +
    ordinal.to_string() +
    "|tokens=" +
    maximum_tokens.to_string() +
    "|ratio=" +
    compression_ratio.to_string() +
    "|maximum-positions=" +
    maximum_position_count.to_string() +
    "|maximum-slots=" +
    maximum_slots_per_row.to_string() +
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
    "/" +
    dimensions.block_x().to_string() +
    ":" +
    dimensions.block_y().to_string() +
    ":" +
    dimensions.block_z().to_string() +
    "/" +
    dimensions.shared_memory_bytes().to_string() +
    "|operands=" +
    compressed_index_operand_text(operands) +
    "|source=" +
    source_digest.as_hex(),
  )
}

///|
/// Lowers exact deterministic complete-slot selection. Inputs are I32
/// `query_positions[maximum_tokens]` and `cache_offsets[maximum_tokens]`.
/// Outputs are I32 `slot_counts[maximum_tokens]` and an I32 dense matrix
/// `[maximum_tokens, floor(maximum_position_count / compression_ratio)]`;
/// each row contains the selected prefix followed by `-1` padding. Runtime
/// `counts[3]` bounds live rows. The candidate remains non-bindable pending
/// offline compilation and physical correctness qualification.
pub fn lower_deterministic_compressed_index_cuda_aot_candidate(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : AdvancedDecoderCudaAotLimits,
) -> AdvancedDecoderCompressedIndexCudaCandidate raise AdvancedDecoderCudaAotError {
  if maximum_tokens <= 0 || maximum_tokens > limits.maximum_tokens {
    raise Invalid(Limit)
  }
  if compiler.reassociate() {
    raise Invalid(Compiler)
  }
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (
      DeterministicCompressedIndex(compression_ratio),
      DeterministicCompressedIndexAot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let maximum_position_count = requirements.geometry().maximum_position_count()
  if compression_ratio <= 1 ||
    maximum_position_count <= 0 ||
    compression_ratio > maximum_position_count {
    raise Invalid(Shape)
  }
  let maximum_slots_per_row = maximum_position_count / compression_ratio
  let row_bytes = checked_mul(maximum_tokens.to_int64(), 4L)
  let index_bytes = checked_mul(
    checked_mul(maximum_tokens.to_int64(), maximum_slots_per_row.to_int64()),
    4L,
  )
  if row_bytes > 2147483647L || index_bytes > 2147483647L {
    raise SizeOverflow
  }
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=maximum_tokens,
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
    operand_with_alignment(StepCountsInput, 20L, 4L),
    operand_with_alignment(CompressedQueryPositionsInput, row_bytes, 4L),
    operand_with_alignment(CompressedCacheOffsetsInput, row_bytes, 4L),
    operand_with_alignment(CompressedIndexCountsOutput, row_bytes, 4L),
    operand_with_alignment(CompressedIndicesOutput, index_bytes, 4L),
  ]
  let function_symbol = "lunaflux_decoder_causal_complete_slots_i32_v1"
  let rendered = @decoder_compressed_index_cuda_source.causal_complete_slots_i32_source(
    compression_ratio~,
    maximum_position_count~,
    maximum_rows=maximum_tokens,
    function_symbol~,
  ) catch {
    error => raise CompressedIndexSource(error)
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = AdvancedDecoderCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let encoded_recipe = compressed_index_recipe(
    requirements, requirement_ordinal, maximum_tokens, compression_ratio, maximum_position_count,
    maximum_slots_per_row, target, compiler, dimensions, operands, source_digest,
  )
  if encoded_recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    family: CausalCompleteSlotsI32,
    numerical_contract: BitExactCausalCompleteSlotsI32V1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    target,
    maximum_tokens,
    compression_ratio,
    maximum_position_count,
    maximum_slots_per_row,
    dimensions,
    operands: ReadOnlyArray::from_array(operands),
    compiler,
    function_symbol,
    source,
    source_digest,
    recipe: encoded_recipe,
    recipe_digest: AdvancedDecoderCudaRecipeDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(encoded_recipe)),
    ),
  }
}
