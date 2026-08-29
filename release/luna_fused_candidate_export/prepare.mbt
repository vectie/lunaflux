///|
fn fused_export_sha256(bytes : Bytes) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
fn fused_export_key(operation : @plan.OperationId, family : String) -> String {
  let value = operation.as_int().to_string()
  "op_" + "0".repeat(8 - value.length()) + value + "_" + family
}

///|
fn fused_export_append(output : StringBuilder, line : String) -> Unit {
  output.write_string(line)
  output.write_string("\n")
}

///|
fn fused_export_reference_candidates(
  set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
) -> (
  @luna_cuda_projection_aot.ProjectionCudaLowering,
  @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  @luna_cuda_paged_attention_aot.PagedAttentionCudaLowering,
  @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  @luna_cuda_pointwise_aot.PointwiseCudaLowering,
) raise LunaFusedCandidateExportError {
  let candidates = set.candidates()
  for index in 0..<(candidates.length() - 2) {
    guard candidates[index] is Projection(qkv) &&
      qkv.family() == QkvProjection &&
      candidates[index + 1] is Pointwise(rope) &&
      rope.family() == PositionedRotary &&
      candidates[index + 2] is PagedAttention(attention) else {
      continue
    }
    for residual_index in (index + 3)..<(candidates.length() - 1) {
      guard candidates[residual_index] is Pointwise(residual) &&
        residual.family() == ResidualAdd &&
        candidates[residual_index + 1] is Pointwise(norm) &&
        norm.family() == RmsNorm else {
        continue
      }
      return (qkv, rope, attention, residual, norm)
    }
  }
  raise FusedCandidateExportFailed(Qualification)
}

///|
fn fused_export_kv_write(
  plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
  attention : @luna_cuda_paged_attention_aot.PagedAttentionCudaLowering,
) -> @luna_cuda_pointwise_aot.PointwiseCudaLowering raise LunaFusedCandidateExportError {
  let operands : Array[@launch_contract.AotOperand] = []
  for operand in attention.operands() {
    match operand.role() {
      ActivationOutput(_) => ()
      _ => operands.push(operand)
    }
  }
  let operation_index = attention.operation_id().as_int()
  guard plan.operations().get(operation_index) is Some(operation) else {
    raise FusedCandidateExportFailed(Qualification)
  }
  @luna_cuda_pointwise_aot.lower_auxiliary_kv_cache_write_cuda_aot(
    operation,
    set.profile(),
    layout,
    @catalog.AotKernelEntryPointId::from_stable_int(2901),
    set.target(),
    set.compiler_policy(),
    operands,
    @luna_cuda_pointwise_aot.PointwiseAotLimits::new(
      max_source_bytes=1048576,
      max_recipe_bytes=65536,
    ),
  ) catch {
    _ => raise FusedCandidateExportFailed(Qualification)
  }
}

///|
fn fused_export_declaration(
  plan : @plan.ModelPlan,
  set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
  qkv : @luna_cuda_fused_parallel_aot.FusedQkvRopeKvWriteCudaCandidate,
  residual_norm : @luna_cuda_fused_parallel_aot.FusedResidualRmsNormCudaCandidate,
  qkv_key : String,
  residual_norm_key : String,
) -> Bytes {
  let identity = plan.identity()
  let profile = set.profile()
  let output = StringBuilder()
  fused_export_append(
    output, "schema=lunaflux-fused-candidate-qualification-set.v1",
  )
  fused_export_append(
    output,
    "model_content_sha256=" + identity.content().as_hex(),
  )
  fused_export_append(output, "model_plan_sha256=" + identity.plan().as_hex())
  fused_export_append(output, "target=sm_120")
  fused_export_append(output, "block_threads=128")
  fused_export_append(output, "profile_id=" + profile.id().as_int().to_string())
  fused_export_append(
    output,
    "max_query_tokens=" + profile.max_query_tokens().to_string(),
  )
  fused_export_append(output, "candidate_count=2")
  fused_export_append(
    output,
    "candidate=" + qkv_key + "," + qkv.candidate_digest().as_hex(),
  )
  fused_export_append(
    output,
    "candidate=" +
    residual_norm_key +
    "," +
    residual_norm.candidate_digest().as_hex(),
  )
  fused_export_append(output, "qualification_only=1")
  fused_export_append(output, "manifest_bindable=0")
  fused_export_append(output, "compiler_authority=0")
  fused_export_append(output, "artifact_authority=0")
  fused_export_append(output, "device_authority=0")
  fused_export_append(output, "runtime_authority=0")
  fused_export_append(output, "promotion_authority=0")
  @utf8.encode(output.to_string())
}

///|
fn fused_export_inventory(
  declaration_sha256 : String,
  qkv : @luna_cuda_fused_parallel_aot.FusedQkvRopeKvWriteCudaCandidate,
  residual_norm : @luna_cuda_fused_parallel_aot.FusedResidualRmsNormCudaCandidate,
  qkv_key : String,
  residual_norm_key : String,
) -> Bytes {
  let output = StringBuilder()
  fused_export_append(output, declaration_sha256 + "  candidate-set.v1")
  fused_export_append(
    output,
    qkv.recipe_digest().as_hex() + "  recipes/" + qkv_key + ".recipe",
  )
  fused_export_append(
    output,
    residual_norm.recipe_digest().as_hex() +
    "  recipes/" +
    residual_norm_key +
    ".recipe",
  )
  fused_export_append(
    output,
    qkv.source_digest().as_hex() + "  sources/" + qkv_key + ".cu",
  )
  fused_export_append(
    output,
    residual_norm.source_digest().as_hex() +
    "  sources/" +
    residual_norm_key +
    ".cu",
  )
  fused_export_append(
    output,
    qkv.qkv_fallback().source_digest().as_hex() +
    "  sources/oracle_qkv_projection.cu",
  )
  fused_export_append(
    output,
    qkv.qkv_fallback().recipe_digest().as_hex() +
    "  recipes/oracle_qkv_projection.recipe",
  )
  fused_export_append(
    output,
    qkv.rope_fallback().source_digest().as_hex() +
    "  sources/oracle_positioned_rope.cu",
  )
  fused_export_append(
    output,
    qkv.rope_fallback().recipe_digest().as_hex() +
    "  recipes/oracle_positioned_rope.recipe",
  )
  fused_export_append(
    output,
    qkv.kv_write_fallback().source_digest().as_hex() +
    "  sources/oracle_paged_kv_write.cu",
  )
  fused_export_append(
    output,
    qkv.kv_write_fallback().recipe_digest().as_hex() +
    "  recipes/oracle_paged_kv_write.recipe",
  )
  @utf8.encode(output.to_string())
}

///|
/// Derives the exact sm120, block-128 qualification pair from the first
/// complete fused layer in a product-owned BF16 candidate set. The caller
/// cannot inject source, recipes, symbols, operands, or family selection.
pub fn prepare_qualification_set(
  plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
) -> LunaFusedCandidateQualificationSet raise LunaFusedCandidateExportError {
  if set.model_identity() != plan.identity() ||
    !layout.is_canonical_for(plan.identity(), plan.kv_cache_geometry()) ||
    set.target().compute_major() != 12 ||
    set.target().compute_minor() != 0 ||
    !set.target().supports_bf16() ||
    set.compiler_policy().fused_multiply_add() ||
    set.compiler_policy().reassociate() {
    raise FusedCandidateExportFailed(Qualification)
  }
  let (qkv_fallback, rope_fallback, attention, residual, norm) = fused_export_reference_candidates(
    set,
  )
  let kv_write = fused_export_kv_write(plan, layout, set, attention)
  let limits = @luna_cuda_fused_parallel_aot.FusedParallelAotLimits::new(
    maximum_source_bytes=1048576,
    maximum_recipe_bytes=65536,
    maximum_promotion_bytes=65536,
  ) catch {
    _ => raise FusedCandidateExportFailed(Qualification)
  }
  let qkv = @luna_cuda_fused_parallel_aot.lower_fused_qkv_rope_paged_kv_write_candidate(
    plan,
    layout,
    qkv_fallback,
    rope_fallback,
    kv_write,
    set.target(),
    set.compiler_policy(),
    limits,
  ) catch {
    _ => raise FusedCandidateExportFailed(Qualification)
  }
  let residual_norm = @luna_cuda_fused_parallel_aot.lower_fused_residual_rmsnorm_candidate(
    plan,
    residual,
    norm,
    set.target(),
    set.compiler_policy(),
    limits,
  ) catch {
    _ => raise FusedCandidateExportFailed(Qualification)
  }
  if qkv.launch_dimensions().block_x() != 128 ||
    residual_norm.launch_dimensions().block_x() != 128 ||
    qkv.manifest_bindable() ||
    residual_norm.manifest_bindable() {
    raise FusedCandidateExportFailed(Qualification)
  }
  let qkv_key = fused_export_key(
    qkv.qkv_fallback().operation_id(),
    "fused_qkv_positioned_rope_paged_kv_write",
  )
  let residual_norm_key = fused_export_key(
    residual_norm.residual_fallback().operation_id(),
    "fused_residual_rmsnorm",
  )
  let declaration = fused_export_declaration(
    plan, set, qkv, residual_norm, qkv_key, residual_norm_key,
  )
  let declaration_sha256 = fused_export_sha256(declaration)
  let inventory = fused_export_inventory(
    declaration_sha256, qkv, residual_norm, qkv_key, residual_norm_key,
  )
  {
    qkv,
    residual_norm,
    qkv_fallback,
    rope_fallback,
    kv_write_fallback: kv_write,
    qkv_key,
    residual_norm_key,
    declaration,
    declaration_sha256,
    inventory,
    inventory_sha256: fused_export_sha256(inventory),
  }
}
