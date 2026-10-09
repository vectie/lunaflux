///|
fn compressed_kv_prepare_operand_text(
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
      PreparedKeyValueModesInput => "modes"
      PreparedKeyValueSequenceLengthsInput => "sequence-lengths"
      UncompressedKeyValueInput => "ordinary-kv"
      CompressedKeyValueInput => "compressed-kv"
      PreparedKeyValueLengthsOutput => "prepared-lengths"
      PreparedKeyValueOutput => "prepared-kv"
      _ => abort("unexpected compressed KV prepare operand")
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
fn compressed_kv_prepare_recipe(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  ordinal : Int,
  maximum_batches : Int,
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
    "lunaflux.advanced-decoder-compressed-kv-prepare-cuda-aot.v1" +
    "|family=non-overlapping-shared-kv-prepare-bf16-v1" +
    "|numerical=bit-exact-bf16-segment-copy-v1" +
    "|ownership=borrowed-no-live-cache" +
    "|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|ordinal=" +
    ordinal.to_string() +
    "|batches=" +
    maximum_batches.to_string() +
    "|geometry=" +
    window_size.to_string() +
    ":" +
    compression_ratio.to_string() +
    ":" +
    requirements.geometry().attention_head_dimension().to_string() +
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
    compressed_kv_prepare_operand_text(operands) +
    "|source=" +
    source_digest.as_hex(),
  )
}

///|
/// Lowers only borrowed-buffer materialization of the official non-overlapping
/// shared K/V layout. Learned compression, mutable compressor state, cache
/// ownership, index construction, sparse attention, and execution remain out.
pub fn lower_non_overlapping_compressed_kv_prepare_cuda_aot_candidate(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_batches~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : AdvancedDecoderCudaAotLimits,
) -> AdvancedDecoderCompressedKvPrepareCudaCandidate raise AdvancedDecoderCudaAotError {
  if maximum_batches <= 0 || maximum_batches > limits.maximum_tokens {
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
      NonOverlappingCompressedKeyValuePrepareV1(window_size, compression_ratio),
      NonOverlappingCompressedKeyValuePrepareV1Aot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let geometry = requirements.geometry()
  let head_dimension = geometry.attention_head_dimension()
  let maximum_position_count = geometry.maximum_position_count()
  if !target.supports_bf16() {
    raise Invalid(Target)
  }
  if window_size <= 0 ||
    compression_ratio <= 1 ||
    window_size > maximum_position_count ||
    compression_ratio > maximum_position_count ||
    head_dimension <= 0 {
    raise Invalid(Shape)
  }
  let maximum_compressed_slots = maximum_position_count / compression_ratio
  let maximum_key_slots = maximum_position_count + maximum_compressed_slots
  let row_bytes = checked_mul(maximum_batches.to_int64(), 4L)
  let ordinary_bytes = checked_mul(
    checked_mul(
      checked_mul(maximum_batches.to_int64(), maximum_position_count.to_int64()),
      head_dimension.to_int64(),
    ),
    2L,
  )
  let compressed_bytes = checked_mul(
    checked_mul(
      checked_mul(
        maximum_batches.to_int64(),
        maximum_compressed_slots.to_int64(),
      ),
      head_dimension.to_int64(),
    ),
    2L,
  )
  let prepared_bytes = checked_mul(
    checked_mul(
      checked_mul(maximum_batches.to_int64(), maximum_key_slots.to_int64()),
      head_dimension.to_int64(),
    ),
    2L,
  )
  if ordinary_bytes > 2147483647L ||
    compressed_bytes > 2147483647L ||
    prepared_bytes > 2147483647L {
    raise SizeOverflow
  }
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=maximum_batches,
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
    operand_with_alignment(PreparedKeyValueModesInput, row_bytes, 4L),
    operand_with_alignment(PreparedKeyValueSequenceLengthsInput, row_bytes, 4L),
    operand_with_alignment(UncompressedKeyValueInput, ordinary_bytes, 2L),
    operand_with_alignment(CompressedKeyValueInput, compressed_bytes, 2L),
    operand_with_alignment(PreparedKeyValueLengthsOutput, row_bytes, 4L),
    operand_with_alignment(PreparedKeyValueOutput, prepared_bytes, 2L),
  ]
  let function_symbol = "lunaflux_decoder_compressed_kv_prepare_bf16_v1"
  let rendered = @decoder_compressed_kv_prepare_cuda_source.non_overlapping_shared_kv_prepare_bf16_source(
    window_size~,
    compression_ratio~,
    head_dimension~,
    maximum_position_count~,
    maximum_batches~,
    function_symbol~,
  ) catch {
    error => raise CompressedKvPrepareSource(error)
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = AdvancedDecoderCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let recipe = compressed_kv_prepare_recipe(
    requirements, requirement_ordinal, maximum_batches, window_size, compression_ratio,
    target, compiler, dimensions, operands, source_digest,
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    family: NonOverlappingSharedKeyValuePrepareBf16V1,
    numerical_contract: BitExactBf16SegmentCopyV1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    target,
    maximum_batches,
    window_size,
    compression_ratio,
    head_dimension,
    maximum_position_count,
    maximum_compressed_slots,
    maximum_key_slots,
    dimensions,
    operands: ReadOnlyArray::from_array(operands),
    compiler,
    function_symbol,
    source,
    source_digest,
    recipe,
    recipe_digest: AdvancedDecoderCudaRecipeDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(recipe)),
    ),
  }
}
