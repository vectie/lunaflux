///|
fn window_compressed_join_operand_text(
  operands : ArrayView[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ],
) -> String {
  let text = StringBuilder()
  for index, operand in operands {
    if index > 0 {
      text.write_char(',')
    }
    let role = match operand.role() {
      StepCountsInput => "step-counts"
      WindowQueryPositionsInput => "query-positions"
      WindowSequenceLengthsInput => "sequence-lengths"
      WindowQueryModesInput => "query-modes"
      CompressedIndexCountsInput => "compressed-counts"
      CompressedIndicesInput => "compressed-indices"
      JoinedIndexCountsOutput => "joined-counts"
      JoinedIndicesOutput => "joined-indices"
      _ => abort("unexpected window-compressed join operand")
    }
    text.write_string(role)
    text.write_char('@')
    text.write_string(operand.byte_count().to_string())
    text.write_char('/')
    text.write_string(operand.alignment().to_string())
  }
  text.to_string()
}

///|
fn window_compressed_join_recipe(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  ordinal : Int,
  maximum_tokens : Int,
  window_size : Int,
  compression_ratio : Int,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  dimensions : @launch_contract.AotLaunchDimensions,
  operands : ArrayView[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ],
  source_digest : AdvancedDecoderCudaSourceDigest,
) -> Bytes {
  let identity = compiler.identity()
  @utf8.encode(
    "lunaflux.advanced-decoder-window-compressed-join-cuda-aot.v1" +
    "|family=deterministic-window-compressed-join-i32-v1" +
    "|numerical=bit-exact-official-window-then-compressed-i32-v1" +
    "|ordering=window-then-compressed-no-sort-no-dedup" +
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
    "|geometry=" +
    window_size.to_string() +
    ":" +
    compression_ratio.to_string() +
    ":" +
    requirements.geometry().maximum_position_count().to_string() +
    "|sm=" +
    target.compute_major().to_string() +
    "." +
    target.compute_minor().to_string() +
    ":" +
    target.supports_bf16().to_string() +
    ":" +
    target.supports_cublas_lt().to_string() +
    "|compiler=" +
    identity.toolchain_digest_sha256() +
    ":" +
    identity.major().to_string() +
    "." +
    identity.minor().to_string() +
    "." +
    identity.patch().to_string() +
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
    ":1:1/1:1:1/0" +
    "|operands=" +
    window_compressed_join_operand_text(operands) +
    "|source=" +
    source_digest.as_hex(),
  )
}

///|
/// Lowers the official window-index generation plus literal compressed-index
/// concatenation boundary. Per-row modes make the official prefill-contiguous
/// and decode-ring branches explicit. No compressed-index production, K/V
/// preparation, attention, or execution authority is included.
pub fn lower_deterministic_window_compressed_join_cuda_aot_candidate(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : AdvancedDecoderCudaAotLimits,
) -> AdvancedDecoderWindowCompressedJoinCudaCandidate raise AdvancedDecoderCudaAotError {
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
      DeterministicWindowCompressedIndexJoinV1(window_size, compression_ratio),
      DeterministicWindowCompressedIndexJoinV1Aot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let maximum_position_count = requirements.geometry().maximum_position_count()
  if window_size <= 0 ||
    compression_ratio <= 1 ||
    window_size > maximum_position_count ||
    compression_ratio > maximum_position_count {
    raise Invalid(Shape)
  }
  let maximum_compressed_width = maximum_position_count / compression_ratio
  let maximum_joined_width = window_size + maximum_compressed_width
  let row_bytes = checked_mul(maximum_tokens.to_int64(), 4L)
  let compressed_bytes = checked_mul(
    checked_mul(maximum_tokens.to_int64(), maximum_compressed_width.to_int64()),
    4L,
  )
  let joined_bytes = checked_mul(
    checked_mul(maximum_tokens.to_int64(), maximum_joined_width.to_int64()),
    4L,
  )
  if row_bytes > 2147483647L ||
    compressed_bytes > 2147483647L ||
    joined_bytes > 2147483647L {
    raise SizeOverflow
  }
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=maximum_tokens,
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
    operand_with_alignment(StepCountsInput, 20L, 4L),
    operand_with_alignment(WindowQueryPositionsInput, row_bytes, 4L),
    operand_with_alignment(WindowSequenceLengthsInput, row_bytes, 4L),
    operand_with_alignment(WindowQueryModesInput, row_bytes, 4L),
    operand_with_alignment(CompressedIndexCountsInput, row_bytes, 4L),
    operand_with_alignment(CompressedIndicesInput, compressed_bytes, 4L),
    operand_with_alignment(JoinedIndexCountsOutput, row_bytes, 4L),
    operand_with_alignment(JoinedIndicesOutput, joined_bytes, 4L),
  ]
  let function_symbol = "lunaflux_decoder_window_compressed_join_i32_v1"
  let rendered = @decoder_window_compressed_join_cuda_source.deterministic_window_compressed_join_i32_source(
    window_size~,
    compression_ratio~,
    maximum_position_count~,
    maximum_rows=maximum_tokens,
    function_symbol~,
  ) catch {
    error => raise WindowCompressedJoinSource(error)
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = AdvancedDecoderCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let encoded_recipe = window_compressed_join_recipe(
    requirements, requirement_ordinal, maximum_tokens, window_size, compression_ratio,
    target, compiler, dimensions, operands, source_digest,
  )
  if encoded_recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    family: DeterministicWindowCompressedJoinI32V1,
    numerical_contract: BitExactOfficialWindowThenCompressedI32V1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    target,
    maximum_tokens,
    window_size,
    compression_ratio,
    maximum_position_count,
    maximum_compressed_width,
    maximum_joined_width,
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
