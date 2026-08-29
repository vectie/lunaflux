///|
struct FusedProbeBindings {
  qkv : @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding
  residual : @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding
}

///|
fn probe_receipt_digest(
  receipt : Bytes,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest raise {
  @luna_cuda_fused_parallel_aot.FusedParallelDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(receipt)),
  )
}

///|
fn bind_fused_probe_artifacts(
  set : @luna_fused_candidate_export.LunaFusedCandidateQualificationSet,
  qkv_first : Bytes,
  qkv_second : Bytes,
  qkv_receipt : Bytes,
  residual_first : Bytes,
  residual_second : Bytes,
  residual_receipt : Bytes,
) -> FusedProbeBindings raise {
  let limits = @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactLimits::new(
    maximum_module_bytes=16777216L,
    maximum_receipt_bytes=65536,
    maximum_binding_bytes=65536,
  )
  let qkv = @luna_cuda_fused_parallel_aot.bind_fused_qkv_rope_kv_compiled_candidate(
    set.qkv_candidate(),
    qkv_first,
    qkv_second,
    qkv_receipt,
    probe_receipt_digest(qkv_receipt),
    limits,
  )
  let residual = @luna_cuda_fused_parallel_aot.bind_fused_residual_rmsnorm_compiled_candidate(
    set.residual_norm_candidate(),
    residual_first,
    residual_second,
    residual_receipt,
    probe_receipt_digest(residual_receipt),
    limits,
  )
  require_fused_probe(
    !qkv.manifest_bindable() && !residual.manifest_bindable(),
    "compiled candidate gained manifest authority",
  )
  require_fused_probe(
    qkv.target().compute_major() == 12 &&
    qkv.target().compute_minor() == 0 &&
    residual.target().compute_major() == 12 &&
    residual.target().compute_minor() == 0,
    "compiled candidate target is not exact sm120",
  )
  { qkv, residual }
}
