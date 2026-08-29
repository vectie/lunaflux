///|
fn fused_physical_digest(
  bytes : Bytes,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  let state = @canonical_sha256.CanonicalSha256::new()
  if !state.update_bytes(bytes) {
    abort("fresh canonical SHA-256 rejected bytes")
  }
  @luna_cuda_fused_parallel_aot.FusedParallelDigest::from_sha256(
    state.finalize_hex(),
  ) catch {
    _ => abort("canonical SHA-256 produced an invalid digest")
  }
}

///|
fn fused_physical_nonzero_digest(
  value : String,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest raise FusedPhysicalEvidenceError {
  @luna_cuda_fused_parallel_aot.FusedParallelDigest::from_sha256(value) catch {
    _ => raise FusedPhysicalEvidenceInvalid(Digest)
  }
}

///|
fn fused_physical_referee_shape_set_digest() -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  fused_physical_digest(
    @utf8.encode(
      "schema=lunaflux-fused-scalar-referee-shape-set.v1\n" +
      "shape_count=4\n" +
      "shape=0,single-origin,0,1\n" +
      "shape=1,single-page-tail,15,2\n" +
      "shape=2,cross-page-pair,15|16,1|3\n" +
      "shape=3,eight-token-page-boundaries,0|1|14|15|16|17|30|31,0|2\n",
    ),
  )
}
