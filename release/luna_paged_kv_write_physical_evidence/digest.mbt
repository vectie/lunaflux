///|
fn kv_physical_sha256(
  bytes : Bytes,
) -> @luna_cuda_paged_kv_write_aot.PagedKvWriteDigest {
  let state = @canonical_sha256.CanonicalSha256::new()
  if !state.update_bytes(bytes) {
    abort("fresh SHA-256 rejected bytes")
  }
  @luna_cuda_paged_kv_write_aot.PagedKvWriteDigest::from_sha256(
    state.finalize_hex(),
  ) catch {
    _ => abort("SHA-256 returned an invalid digest")
  }
}
