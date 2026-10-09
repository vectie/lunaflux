///|
/// One request's BF16 packed input and immutable row mapping in one allocation.
/// Upload the mapping before any frame and keep it immutable while queued.
pub struct PackedHidden {
  priv plan : @packed_rows_plan.Plan
  priv allocation : @device.Allocation
  priv output_bytes : Int64
  priv mapping_offset : Int64
}

///|
pub fn PackedHidden::allocate(
  context : @device.Context,
  plan : @packed_rows_plan.Plan,
) -> Self raise @device.DeviceError {
  guard plan.element_count() <= 2147483647L else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  let output_bytes = plan.element_count() * 2L
  let mapping_offset = output_bytes + (256L - output_bytes % 256L) % 256L
  let allocation = context.allocate(
    mapping_offset + plan.mapping().length().to_int64(),
  )
  { plan, allocation, output_bytes, mapping_offset, }
}

///|
pub fn PackedHidden::upload_mapping(
  self : Self,
) -> Unit raise @device.DeviceError {
  let bytes = self.plan.mapping()
  self.allocation.copy_from_host(
    bytes,
    0L,
    self.mapping_offset,
    bytes.length().to_int64(),
  )
}

///|
pub fn PackedHidden::output_argument(
  self : Self,
) -> @device.KernelArgument raise @device.DeviceError {
  @device.KernelArgument::device_region(
    self.allocation,
    offset=0L,
    byte_count=self.output_bytes,
    alignment=256L,
  )
}

///|
/// Startup-only bridge to consumers which bind allocation regions. The source
/// must use this exact row map and width. The callback may prepare queue leases,
/// but must not close the allocation or modify the trailing immutable mapping.
pub fn[T] PackedHidden::with_state(
  self : Self,
  expected_mapping : Bytes,
  width~ : Int,
  bind : (@device.Allocation, Int64) -> T raise,
) -> T raise {
  guard width == self.plan.width() && expected_mapping == self.plan.mapping() else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  bind(self.allocation, 0L)
}

///|
/// Startup binding. Function must match this map's width/extent/block size.
/// Inputs must already be projected BF16 rows. The map encodes row order and
/// padding; this layer never assumes conditioning/video/audio concatenation.
pub fn PackedHidden::launch(
  self : Self,
  function : @device.Function,
  projections : LatentProjections,
  conditioning~ : @device.Allocation?,
  conditioning_offset~ : Int64,
  block_size~ : Int,
) -> @device.OrderedKernelLaunch raise @device.DeviceError {
  for
    (latent, source) in [
      (
        @joint_diffusion_plan.JointDiffusionLatentKind::VideoLatent,
        @packed_rows_plan.Source::Video,
      ),
      (AudioLatent, Audio),
    ] {
    let shape = projections.plan.shape(latent)
    guard shape.rows == self.plan.input_rows(source).to_int64() &&
      shape.output_width == self.plan.width() else {
      raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
    }
  }
  let condition_rows = self.plan.input_rows(Conditioning)
  let condition_argument = if condition_rows == 0 {
    // Unread ABI placeholder. No mapping row selects this input.
    self.output_argument()
  } else {
    guard conditioning is Some(allocation) else {
      raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
    }
    @device.KernelArgument::device_region(
      allocation,
      offset=conditioning_offset,
      byte_count=condition_rows.to_int64() * self.plan.width().to_int64() * 2L,
      alignment=2L,
    )
  }
  let arguments = @device.KernelArguments::new([
    condition_argument,
    projections.output_argument(VideoLatent),
    projections.output_argument(AudioLatent),
    @device.KernelArgument::device_region(
      self.allocation,
      offset=self.mapping_offset,
      byte_count=self.plan.mapping().length().to_int64(),
      alignment=256L,
    ),
    self.output_argument(),
  ])
  @device.OrderedKernelLaunch::new(
    function,
    linear_dimensions(self.plan.element_count(), block_size),
    arguments,
  )
}

///|
/// Release after all queues; a failed close retains ownership for retry.
pub fn PackedHidden::close(self : Self) -> Unit raise @device.DeviceError {
  self.allocation.close()
}
