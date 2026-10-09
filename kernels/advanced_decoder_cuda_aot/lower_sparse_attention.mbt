///|
fn sparse_attention_operand_text(
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
      SparseAttentionQueryInput => "query"
      SparseAttentionQueryBatchIdsInput => "query-batch-ids"
      SparseAttentionKeyValueInput => "shared-key-value"
      SparseAttentionCacheLengthsInput => "cache-lengths"
      SparseAttentionSinkInput => "attention-sink"
      SparseAttentionSelectedIndicesInput => "selected-indices"
      SparseAttentionOutput => "output"
      _ => abort("unexpected selected sparse-attention operand")
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
fn sparse_attention_recipe(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  ordinal : Int,
  maximum_tokens : Int,
  maximum_batches : Int,
  window_size : Int,
  compression_ratio : Int,
  selected_width : Int,
  maximum_key_slots : Int,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  dimensions : @launch_contract.AotLaunchDimensions,
  operands : ArrayView[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ],
  source_digest : AdvancedDecoderCudaSourceDigest,
) -> Bytes {
  let geometry = requirements.geometry()
  let compiler_identity = compiler.identity()
  @utf8.encode(
    "lunaflux.advanced-decoder-selected-sparse-attention-cuda-aot.v1" +
    "|family=shared-key-value-selected-attention-bf16" +
    "|numerical=serial-ordered-f32-sink-denominator-bf16-output-v1" +
    "|scope=joined-indices-and-borrowed-prepared-kv-only" +
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
    "|batches=" +
    maximum_batches.to_string() +
    "|geometry=" +
    geometry.attention_head_count().to_string() +
    ":" +
    geometry.attention_head_dimension().to_string() +
    ":" +
    geometry.maximum_position_count().to_string() +
    ":" +
    window_size.to_string() +
    ":" +
    compression_ratio.to_string() +
    ":" +
    selected_width.to_string() +
    ":" +
    maximum_key_slots.to_string() +
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
    ":1:1/1:1:1/0" +
    "|operands=" +
    sparse_attention_operand_text(operands) +
    "|source=" +
    source_digest.as_hex(),
  )
}

///|
/// Lowers the selected-attention phase only. Selected indices must already
/// contain the non-overlapping window+compressed join, and K/V is a borrowed,
/// already-prepared shared buffer. This source object remains non-bindable.
pub fn lower_selected_sparse_attention_cuda_aot_candidate(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  maximum_batches~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : AdvancedDecoderCudaAotLimits,
) -> AdvancedDecoderSparseAttentionCudaCandidate raise AdvancedDecoderCudaAotError {
  if maximum_tokens <= 0 ||
    maximum_tokens > limits.maximum_tokens ||
    maximum_batches <= 0 ||
    maximum_batches > maximum_tokens {
    raise Invalid(Limit)
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
  guard (requirement.operation(), requirement.capability())
    is (
      SelectedSparseAttentionV1(window_size, compression_ratio),
      SelectedSparseAttentionV1Aot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let geometry = requirements.geometry()
  let query_head_count = geometry.attention_head_count()
  let head_dimension = geometry.attention_head_dimension()
  let maximum_position_count = geometry.maximum_position_count()
  if window_size <= 0 ||
    compression_ratio <= 1 ||
    window_size > maximum_position_count ||
    compression_ratio > maximum_position_count ||
    query_head_count <= 0 ||
    head_dimension <= 0 {
    raise Invalid(Shape)
  }
  let selected_width = window_size + maximum_position_count / compression_ratio
  let maximum_key_slots = maximum_position_count +
    maximum_position_count / compression_ratio
  let rows = maximum_tokens.to_int64()
  let heads = query_head_count.to_int64()
  let width = head_dimension.to_int64()
  let query_bytes = checked_mul(
    checked_mul(checked_mul(rows, heads), width),
    2L,
  )
  let key_value_bytes = checked_mul(
    checked_mul(
      checked_mul(maximum_batches.to_int64(), maximum_key_slots.to_int64()),
      width,
    ),
    2L,
  )
  let row_i32_bytes = checked_mul(rows, 4L)
  let selected_bytes = checked_mul(
    checked_mul(rows, selected_width.to_int64()),
    4L,
  )
  let cache_length_bytes = checked_mul(maximum_batches.to_int64(), 4L)
  let sink_bytes = checked_mul(heads, 4L)
  for
    byte_count in [
      query_bytes, key_value_bytes, row_i32_bytes, selected_bytes, cache_length_bytes,
      sink_bytes,
    ] {
    if byte_count > 2147483647L {
      raise SizeOverflow
    }
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
    operand_with_alignment(SparseAttentionQueryInput, query_bytes, 16L),
    operand_with_alignment(SparseAttentionQueryBatchIdsInput, row_i32_bytes, 4L),
    operand_with_alignment(SparseAttentionKeyValueInput, key_value_bytes, 16L),
    operand_with_alignment(
      SparseAttentionCacheLengthsInput,
      cache_length_bytes,
      4L,
    ),
    operand_with_alignment(SparseAttentionSinkInput, sink_bytes, 4L),
    operand_with_alignment(
      SparseAttentionSelectedIndicesInput,
      selected_bytes,
      4L,
    ),
    operand_with_alignment(SparseAttentionOutput, query_bytes, 16L),
  ]
  let function_symbol = "lunaflux_decoder_selected_sparse_attention_bf16_v1"
  let rendered = @decoder_selected_sparse_attention_cuda_source.shared_key_value_selected_attention_bf16_source(
    query_head_count~,
    head_dimension~,
    selected_width~,
    maximum_key_slots~,
    maximum_rows=maximum_tokens,
    maximum_batches~,
    function_symbol~,
  ) catch {
    error => raise SparseAttentionSource(error)
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = AdvancedDecoderCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let encoded_recipe = sparse_attention_recipe(
    requirements, requirement_ordinal, maximum_tokens, maximum_batches, window_size,
    compression_ratio, selected_width, maximum_key_slots, target, compiler, dimensions,
    operands, source_digest,
  )
  if encoded_recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    family: SharedKeyValueSelectedAttentionBf16,
    numerical_contract: SerialOrderedF32SinkDenominatorBf16OutputV1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    target,
    maximum_tokens,
    maximum_batches,
    query_head_count,
    head_dimension,
    window_size,
    compression_ratio,
    selected_width,
    maximum_key_slots,
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
