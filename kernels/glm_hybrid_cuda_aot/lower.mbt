///|
fn checked_mul(
  left : Int64,
  right : Int64,
) -> Int64 raise GlmHybridCudaAotError {
  if left <= 0L || right <= 0L || left > 9223372036854775807L / right {
    raise SizeOverflow
  }
  left * right
}

///|
fn operand(
  role : GlmHybridFoundationOperandRole,
  byte_count : Int64,
  alignment? : Int64 = 16L,
) -> GlmHybridFoundationOperand raise GlmHybridCudaAotError {
  if byte_count <= 0L ||
    byte_count > 2147483647L ||
    alignment <= 0L ||
    alignment > 4096L ||
    (alignment & (alignment - 1L)) != 0L {
    raise SizeOverflow
  }
  { role, byte_count, alignment, }
}

///|
fn family_tag(family : GlmHybridFoundationAotFamily) -> String {
  match family {
    TokenEmbeddingBf16 => "token-embedding-bf16"
    DecoderRmsNormBf16 => "decoder-rms-norm-bf16"
    FinalRmsNormBf16 => "final-rms-norm-bf16"
    DsaQueryARmsNormBf16 => "dsa-query-a-rms-norm-bf16"
  }
}

///|
fn numerical_tag(
  contract : @decoder_foundation_cuda_source.DecoderFoundationNumericalContract,
) -> String {
  match contract {
    BitExactBf16Copy => "bit-exact-bf16-copy"
    OrderedFloat32ReductionBf16RoundV1 => "ordered-f32-reduction-bf16-round-v1"
  }
}

///|
fn operand_tag(role : GlmHybridFoundationOperandRole) -> String {
  match role {
    StepCountsInput => "step-counts"
    TokenIdsInput => "token-ids"
    HiddenInput => "hidden-input"
    ModelWeights => "model-weights"
    HiddenOutput => "hidden-output"
    DsaQueryCompressedInput => "dsa-query-compressed-input"
    DsaQueryNormalizedOutput => "dsa-query-normalized-output"
  }
}

///|
fn recipe(
  family : GlmHybridFoundationAotFamily,
  numerical_contract : @decoder_foundation_cuda_source.DecoderFoundationNumericalContract,
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  ordinal : Int,
  maximum_tokens : Int,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  dimensions : @launch_contract.AotLaunchDimensions,
  operands : ArrayView[GlmHybridFoundationOperand],
  source_digest : GlmHybridCudaSourceDigest,
) -> Bytes {
  let compiler_identity = compiler.identity()
  let operand_text = StringBuilder()
  for index, item in operands {
    if index > 0 {
      operand_text.write_char(',')
    }
    operand_text.write_string(operand_tag(item.role))
    operand_text.write_char('@')
    operand_text.write_string(item.byte_count.to_string())
    operand_text.write_char('/')
    operand_text.write_string(item.alignment.to_string())
  }
  let geometry = requirements.geometry()
  let stage_contract = match family {
    DsaQueryARmsNormBf16 =>
      "|stage=q-a-rmsnorm-only|input=query-compressed|output=query-normalized" +
      "|excluded=q-b-rope"
    _ => ""
  }
  @utf8.encode(
    "lunaflux.glm-hybrid-cuda-aot.v1" +
    "|family=" +
    family_tag(family) +
    "|numerical=" +
    numerical_tag(numerical_contract) +
    stage_contract +
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
    geometry.vocabulary_size().to_string() +
    ":" +
    geometry.hidden_size().to_string() +
    ":" +
    geometry.decoder_layer_count().to_string() +
    ":" +
    geometry.context_length().to_string() +
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
fn render_embedding(
  vocabulary_size : Int,
  hidden_size : Int,
  symbol : String,
) -> @decoder_foundation_cuda_source.DecoderFoundationCudaSource raise GlmHybridCudaAotError {
  @decoder_foundation_cuda_source.token_embedding_bf16_source(
    vocabulary_size~,
    hidden_size~,
    function_symbol=symbol,
  ) catch {
    InvalidVocabularySize => raise Source(InvalidVocabularySize)
    InvalidHiddenSize => raise Source(InvalidHiddenSize)
    InvalidEpsilon => raise Source(InvalidEpsilon)
    InvalidRowCount => raise Source(InvalidRowCount)
    InvalidFunctionSymbol => raise Source(InvalidFunctionSymbol)
  }
}

///|
fn render_norm(
  hidden_size : Int,
  epsilon : Double,
  symbol : String,
) -> @decoder_foundation_cuda_source.DecoderFoundationCudaSource raise GlmHybridCudaAotError {
  @decoder_foundation_cuda_source.rms_norm_bf16_source(
    hidden_size~,
    epsilon~,
    function_symbol=symbol,
  ) catch {
    InvalidHiddenSize => raise Source(InvalidHiddenSize)
    InvalidEpsilon => raise Source(InvalidEpsilon)
    InvalidFunctionSymbol => raise Source(InvalidFunctionSymbol)
    _ => raise Invalid(Requirement)
  }
}

///|
/// Lowers the shared GLM embedding, decoder RMSNorm, and final RMSNorm subset.
pub fn lower_foundation_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridFoundationCudaCandidate raise GlmHybridCudaAotError {
  if maximum_tokens <= 0 || maximum_tokens > limits.maximum_tokens {
    raise Invalid(Limit)
  }
  if !target.supports_bf16() {
    raise Invalid(Target)
  }
  if compiler.reassociate() {
    raise Invalid(Compiler)
  }
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let geometry = requirements.geometry()
  let width = geometry.hidden_size()
  let activation_bytes = checked_mul(
    checked_mul(maximum_tokens.to_int64(), width.to_int64()),
    2L,
  )
  let (family, rendered, dimensions, operands) = match
    (requirement.operation(), requirement.capability()) {
    (TokenEmbedding, Advanced(EmbeddingLookupAot)) => {
      let work_items = checked_mul(maximum_tokens.to_int64(), width.to_int64())
      let grid = (work_items + 255L) / 256L
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
      let table_bytes = checked_mul(
        checked_mul(geometry.vocabulary_size().to_int64(), width.to_int64()),
        2L,
      )
      let operands = [
        operand(StepCountsInput, 20L, alignment=4L),
        operand(
          TokenIdsInput,
          checked_mul(maximum_tokens.to_int64(), 4L),
          alignment=4L,
        ),
        operand(ModelWeights, table_bytes),
        operand(HiddenOutput, activation_bytes),
      ]
      let symbol = "lunaflux_glm_hybrid_embedding_bf16_v1"
      (
        TokenEmbeddingBf16,
        render_embedding(geometry.vocabulary_size(), width, symbol),
        dimensions,
        operands,
      )
    }
    (DecoderRmsNorm(operation_width, epsilon), DecoderRmsNormAot) => {
      if operation_width != width {
        raise Invalid(Shape)
      }
      let dimensions = @launch_contract.AotLaunchDimensions::new(
        grid_x=maximum_tokens,
        grid_y=1,
        grid_z=1,
        block_x=256,
        block_y=1,
        block_z=1,
        shared_memory_bytes=1024,
      ) catch {
        error => raise Launch(error)
      }
      let operands = [
        operand(StepCountsInput, 20L, alignment=4L),
        operand(HiddenInput, activation_bytes),
        operand(ModelWeights, checked_mul(width.to_int64(), 2L)),
        operand(HiddenOutput, activation_bytes),
      ]
      let symbol = "lunaflux_glm_hybrid_decoder_rms_norm_bf16_v1"
      (
        DecoderRmsNormBf16,
        render_norm(width, epsilon, symbol),
        dimensions,
        operands,
      )
    }
    (FinalRmsNorm(operation_width, epsilon), Advanced(RmsNormAot)) => {
      if operation_width != width {
        raise Invalid(Shape)
      }
      let dimensions = @launch_contract.AotLaunchDimensions::new(
        grid_x=maximum_tokens,
        grid_y=1,
        grid_z=1,
        block_x=256,
        block_y=1,
        block_z=1,
        shared_memory_bytes=1024,
      ) catch {
        error => raise Launch(error)
      }
      let operands = [
        operand(StepCountsInput, 20L, alignment=4L),
        operand(HiddenInput, activation_bytes),
        operand(ModelWeights, checked_mul(width.to_int64(), 2L)),
        operand(HiddenOutput, activation_bytes),
      ]
      let symbol = "lunaflux_glm_hybrid_final_rms_norm_bf16_v1"
      (
        FinalRmsNormBf16,
        render_norm(width, epsilon, symbol),
        dimensions,
        operands,
      )
    }
    (DsaQueryARmsNorm(operation_width, epsilon), DsaQueryARmsNormAot) => {
      if !((width == 6144 && operation_width == 2048) ||
        (width == 4096 && operation_width == 1536)) ||
        epsilon != 0.00001 {
        raise Invalid(Shape)
      }
      let operation_bytes = checked_mul(
        checked_mul(maximum_tokens.to_int64(), operation_width.to_int64()),
        2L,
      )
      let dimensions = @launch_contract.AotLaunchDimensions::new(
        grid_x=maximum_tokens,
        grid_y=1,
        grid_z=1,
        block_x=256,
        block_y=1,
        block_z=1,
        shared_memory_bytes=1024,
      ) catch {
        error => raise Launch(error)
      }
      let operands = [
        operand(StepCountsInput, 20L, alignment=4L),
        operand(DsaQueryCompressedInput, operation_bytes),
        operand(ModelWeights, checked_mul(operation_width.to_int64(), 2L)),
        operand(DsaQueryNormalizedOutput, operation_bytes),
      ]
      let symbol = "lunaflux_glm_hybrid_dsa_query_a_rms_norm_bf16_v1"
      (
        DsaQueryARmsNormBf16,
        render_norm(operation_width, epsilon, symbol),
        dimensions,
        operands,
      )
    }
    _ => raise UnsupportedRequirement(requirement_ordinal)
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = GlmHybridCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let encoded_recipe = recipe(
    family,
    rendered.numerical_contract(),
    requirements,
    requirement_ordinal,
    maximum_tokens,
    target,
    compiler,
    dimensions,
    operands,
    source_digest,
  )
  if encoded_recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    family,
    numerical_contract: rendered.numerical_contract(),
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    target,
    maximum_tokens,
    dimensions,
    operands: ReadOnlyArray::from_array(operands),
    compiler,
    function_symbol: rendered.function_symbol(),
    source,
    source_digest,
    recipe: encoded_recipe,
    recipe_digest: GlmHybridCudaRecipeDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(encoded_recipe)),
    ),
  }
}
