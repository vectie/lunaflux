///|
/// Shared transient storage for serialized joint transformer blocks. Residual
/// state and weights are external. Attention and MLP temporaries overlap only
/// because their producer/consumer intervals do not overlap in the block list.
pub struct BlockWorkspaceLayout {
  priv rows : Int
  priv hidden : Int
  priv heads : Int
  priv dimension : Int
  priv intermediate : Int
  hidden_bytes : Int64
  normalized_offset : Int64
  branch_offset : Int64
  scratch_offset : Int64
  scratch_bytes : Int64
  byte_count : Int64
}

///|
fn aligned_block_bytes(bytes : Int64) -> Int64 {
  bytes + (256L - bytes % 256L) % 256L
}

///|
pub fn BlockWorkspaceLayout::new(
  rows~ : Int,
  hidden_width~ : Int,
  head_count~ : Int,
  head_dimension~ : Int,
  intermediate_width~ : Int,
  max_bytes~ : Int64,
) -> Self raise @device.DeviceError {
  let mlp = StagedMlpLayout::new(rows~, hidden_width~, intermediate_width~)
  let attention = staged_attention_scratch_bytes(
    rows~,
    head_count~,
    head_dimension~,
  )
  let hidden_stride = aligned_block_bytes(mlp.input_bytes)
  let scratch_bytes = Int64::max(attention, mlp.product_bytes)
  let scratch_offset = hidden_stride * 2L
  let byte_count = scratch_offset + scratch_bytes
  guard max_bytes > 0L && byte_count <= max_bytes else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  {
    rows,
    hidden: hidden_width,
    heads: head_count,
    dimension: head_dimension,
    intermediate: intermediate_width,
    hidden_bytes: mlp.input_bytes,
    normalized_offset: 0L,
    branch_offset: hidden_stride,
    scratch_offset,
    scratch_bytes,
    byte_count,
  }
}

///|
pub struct BlockWorkspace {
  priv layout : BlockWorkspaceLayout
  priv allocation : @device.Allocation
}

///|
pub fn BlockWorkspace::allocate(
  context : @device.Context,
  layout : BlockWorkspaceLayout,
) -> Self raise @device.DeviceError {
  { layout, allocation: context.allocate(layout.byte_count), }
}

///|
pub fn BlockWorkspace::layout(self : Self) -> BlockWorkspaceLayout {
  self.layout
}

///|
/// Close after all frame queues have released their leases; failed close may
/// be retried. This workspace does not own residual states or model weights.
pub fn BlockWorkspace::close(self : Self) -> Unit raise @device.DeviceError {
  self.allocation.close()
}
