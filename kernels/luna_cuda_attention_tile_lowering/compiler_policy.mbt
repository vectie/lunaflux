///|
/// Query-owned scores and output folds are intentionally register resident.
/// A cap inherited from the shared-exchange kernel spills those values. Allow
/// the CUDA allocator its architectural ceiling; other kernels keep the
/// caller's existing policy. This is an offline backend decision, not dispatch.
pub fn CudaAttentionTileLowering::compiler_register_ceiling(
  self : CudaAttentionTileLowering,
  existing_ceiling : Int,
) -> Int {
  if self.query_owned_fragments() {
    255
  } else {
    existing_ceiling
  }
}
