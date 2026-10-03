///|
/// Bind the immutable physical address lifetime to this workgroup's vector
/// ownership. CUDA chooses an aligned transfer width; the ownership program
/// itself has no hardware names, subgroup widths or address-space operations.
pub fn CudaAttentionTileLowering::decode_copy_retention(
  self : CudaAttentionTileLowering,
) -> @attention_physical_ir.RetainedCopyOwnership? {
  if self.phase != Decode ||
    self.pipeline_stages != 2 ||
    self.head_dimension % 4 != 0 {
    None
  } else {
    @attention_physical_ir.RetainedCopyOwnership::new(
      vectors=self.key_value_tile *
        (self.head_dimension / self.decode_copy_elements()),
      owners=self.block_threads,
    )
  }
}

///|
/// The blockwise ABI has contiguous, sixteen-byte-aligned head rows. Widen
/// immutable transport, not the arithmetic fold or shared storage lifetime.
pub fn CudaAttentionTileLowering::decode_copy_elements(
  self : CudaAttentionTileLowering,
) -> Int {
  if (
      self.compute == CudaGroupedQueryBlockwiseShuffleDot ||
      self.compute == CudaMatrixQkPvMultiplyAccumulate
    ) &&
    self.head_dimension % 8 == 0 {
    8
  } else {
    4
  }
}
