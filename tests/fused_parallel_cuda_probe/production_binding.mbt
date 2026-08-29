///|
struct FusedProductionProbeBindings {
  qkv : @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding
  residual : @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding
  readonly_module : Bytes
}

///|
fn bind_fused_production_probe_artifacts(
  candidates : FusedProductionProbeCandidates,
  qkv_first : Bytes,
  qkv_second : Bytes,
  qkv_receipt : Bytes,
  readonly_first : Bytes,
  readonly_second : Bytes,
  residual_first : Bytes,
  residual_second : Bytes,
  residual_receipt : Bytes,
) -> FusedProductionProbeBindings raise {
  let limits = @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactLimits::new(
    maximum_module_bytes=16777216L,
    maximum_receipt_bytes=65536,
    maximum_binding_bytes=65536,
  )
  let qkv = @luna_cuda_fused_parallel_aot.bind_fused_qkv_rope_paged_kv_write_production_compiled_candidate(
    candidates.qkv,
    qkv_first,
    qkv_second,
    qkv_receipt,
    probe_receipt_digest(qkv_receipt),
    limits,
  )
  let residual = @luna_cuda_fused_parallel_aot.bind_fused_residual_rmsnorm_production_compiled_candidate(
    candidates.residual,
    residual_first,
    residual_second,
    residual_receipt,
    probe_receipt_digest(residual_receipt),
    limits,
  )
  require_fused_probe(
    !readonly_first.is_empty() && readonly_first == readonly_second,
    "production readonly CUBIN is empty or nondeterministic",
  )
  require_fused_probe(
    !qkv.manifest_bindable() && !residual.manifest_bindable(),
    "production compiled binding gained manifest authority",
  )
  { qkv, residual, readonly_module: readonly_first }
}
