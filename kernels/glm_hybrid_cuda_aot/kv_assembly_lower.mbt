///|
fn has_kv_assembly_producers(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  heads : Int,
  nope : Int,
  rope : Int,
  value : Int,
) -> Bool {
  let mut packed = false
  let mut rotary = false
  for item in requirements.requirements() {
    match item.operation() {
      KeyValueLowRankProjection(512, h, n, v) =>
        if h == heads && n == nope && v == value {
          packed = true
        }
      GlmRotaryPosition(theta, interleaved) =>
        if (rope == 64 && theta is Some(8000000.0) && interleaved) ||
          (rope == 0 && theta is None && !interleaved) {
          rotary = true
        }
      _ => ()
    }
  }
  packed && rotary
}

///|
pub fn lower_dsa_kv_attention_assembly_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridKvAssemblyCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (
      DsaKeyValueAttentionAssembly(heads, nope, rope, value),
      DsaKeyValueAttentionAssemblyAot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let (family, symbol) = match (heads, nope, rope, value) {
    (64, 192, 64, 256) =>
      (
        FullSharedRopeSuffixBroadcast,
        "lunaflux_glm_hybrid_dsa_kv_assembly_rope_bf16_v1",
      )
    (64, 256, 0, 256) =>
      (FlashNoRopeSplit, "lunaflux_glm_hybrid_dsa_kv_assembly_no_rope_bf16_v1")
    _ => raise Invalid(Shape)
  }
  if !has_kv_assembly_producers(requirements, heads, nope, rope, value) {
    raise Invalid(Requirement)
  }
  if maximum_tokens <= 0 ||
    maximum_tokens > limits.maximum_tokens ||
    maximum_tokens > requirements.geometry().context_length() {
    raise Invalid(Limit)
  }
  if !target.supports_bf16() {
    raise Invalid(Target)
  }
  if compiler.reassociate() || compiler.fused_multiply_add() {
    raise Invalid(Compiler)
  }
  let rows = maximum_tokens.to_int64()
  let head_rows = checked_mul(rows, heads.to_int64())
  let packed_bytes = checked_mul(
    checked_mul(head_rows, (nope + value).to_int64()),
    2L,
  )
  let key_cells = checked_mul(head_rows, (nope + rope).to_int64())
  let key_bytes = checked_mul(key_cells, 2L)
  let value_cells = checked_mul(head_rows, value.to_int64())
  let value_bytes = checked_mul(value_cells, 2L)
  let work = checked_mul(
    rows,
    heads.to_int64() * (nope + rope + value).to_int64(),
  )
  let grid = (work + 255L) / 256L
  if grid <= 0L || grid > 2147483647L {
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
  let operands = [
    dense_projection_operand(StepCountsInput, 20L, 4L),
    dense_projection_operand(DsaKeyValueHeadsOutput, packed_bytes, 16L),
  ]
  if rope > 0 {
    operands.push(
      dense_projection_operand(
        DsaAttentionKeyRopeInput,
        checked_mul(checked_mul(rows, rope.to_int64()), 2L),
        16L,
      ),
    )
  }
  operands.push(dense_projection_operand(DsaAttentionKeysInput, key_bytes, 16L))
  operands.push(
    dense_projection_operand(DsaAttentionValuesInput, value_bytes, 16L),
  )
  let rendered = @decoder_kv_assembly_cuda_source.decoder_kv_assembly_bf16_source(
    head_count=heads,
    key_nope_width=nope,
    shared_key_suffix_width=rope,
    value_width=value,
    maximum_rows=maximum_tokens,
    function_symbol=symbol,
  ) catch {
    error => raise KvAssemblySource(error)
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = GlmHybridCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let recipe = @utf8.encode(
    "lunaflux.glm-hybrid-dsa-kv-assembly-cuda-aot.v1" +
    "|layout=packed-token-head-key-nope-then-value" +
    "|key-output=token-head-key-nope-then-shared-rope-broadcast" +
    "|value-output=token-head-value-contiguous" +
    "|copy=bf16-bit-exact-no-arithmetic-no-rounding" +
    "|alias=outputs-distinct-from-inputs-and-each-other" +
    "|qualification=correctness-source-only-non-bindable" +
    "|excluded=kv-projection-rope-cache-state-materialization-upload-launch" +
    "|geometry=" +
    heads.to_string() +
    ":" +
    nope.to_string() +
    ":" +
    rope.to_string() +
    ":" +
    value.to_string() +
    ":" +
    maximum_tokens.to_string() +
    "|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|ordinal=" +
    requirement_ordinal.to_string() +
    "|source=" +
    source_digest.as_hex(),
  )
  if recipe.length() > limits.maximum_recipe_bytes {
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
    maximum_tokens,
    target,
    dimensions,
    operands: ReadOnlyArray::from_array(operands),
    compiler,
    function_symbol: rendered.function_symbol(),
    source,
    source_digest,
    recipe,
    recipe_digest: GlmHybridCudaRecipeDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(recipe)),
    ),
  }
}
