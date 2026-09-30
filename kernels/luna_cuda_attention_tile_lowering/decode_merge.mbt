///|
/// The component-independent partition statistics remain a pure physical
/// program. The CUDA renderer only realizes its single-owner publication.
pub fn CudaAttentionTileLowering::decode_partition_merge(
  self : CudaAttentionTileLowering,
) -> @attention_physical_ir.PartitionMerge? {
  if self.phase == Decode {
    @attention_physical_ir.PartitionMerge::new(
      partitions=self.key_value_partitions,
      dimensions=self.head_dimension,
    )
  } else {
    None
  }
}
