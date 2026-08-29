///|
struct FusedProductionProbeCandidates {
  qkv : @luna_cuda_fused_parallel_aot.FusedQkvRopeKvWriteProductionCudaCandidate
  residual : @luna_cuda_fused_parallel_aot.FusedResidualRmsNormProductionCudaCandidate
  readonly_candidate : @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyProductionCudaCandidate
  oracle_attention : @luna_cuda_paged_attention_aot.PagedAttentionCudaLowering
}

///|
fn fused_production_probe_candidates(
  toolchain_sha256 : String,
) -> (FusedProbeCandidateContext, FusedProductionProbeCandidates) raise {
  let context = fused_probe_candidate_context(toolchain_sha256)
  let limits = @luna_cuda_fused_parallel_aot.FusedParallelAotLimits::new(
    maximum_source_bytes=1048576,
    maximum_recipe_bytes=65536,
    maximum_promotion_bytes=65536,
  )
  let readonly_limits = @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyAotLimits::new(
    maximum_source_bytes=1048576,
    maximum_recipe_bytes=65536,
    maximum_canonical_bytes=65536,
  )
  let readonly_qualification = @luna_cuda_paged_attention_readonly_aot.lower_paged_attention_readonly_cuda_aot_candidate(
    context.attention,
    @catalog.AotKernelEntryPointId::from_stable_int(4901),
    readonly_limits,
  )
  let candidates = {
    qkv: @luna_cuda_fused_parallel_aot.lower_fused_qkv_rope_paged_kv_write_production_candidate(
      context.qualification.qkv_candidate(),
      limits,
    ),
    residual: @luna_cuda_fused_parallel_aot.lower_fused_residual_rmsnorm_production_candidate(
      context.qualification.residual_norm_candidate(),
      limits,
    ),
    readonly_candidate: @luna_cuda_paged_attention_readonly_aot.lower_paged_attention_readonly_production_cuda_aot_candidate(
      readonly_qualification, readonly_limits,
    ),
    oracle_attention: context.attention,
  }
  require_fused_probe(
    !candidates.qkv.manifest_bindable() &&
    !candidates.residual.manifest_bindable() &&
    !candidates.readonly_candidate.manifest_bindable(),
    "production probe candidate gained manifest authority",
  )
  (context, candidates)
}
