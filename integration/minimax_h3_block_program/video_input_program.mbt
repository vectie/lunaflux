///|
/// Request-local final-video bridge. Borrows immutable AOT functions and 24
/// F32 means/stds. Its consuming executor must close before this buffer owner.
pub struct VideoInputProgram {
  priv source : @minimax_h3_block_source.VideoInputSource
  priv layout : @joint_diffusion_device.VaeInputLayout
  priv storage : @device.Allocation
  priv permutation : @device.Function
  priv destandardize : @device.Function
  priv standard_deviations : @device.Allocation
  priv means : @device.Allocation
  priv mut closed : Bool
}

///|
pub fn VideoInputProgram::allocate(
  context : @device.Context,
  source : @minimax_h3_block_source.VideoInputSource,
  permutation : @device.Function,
  destandardize : @device.Function,
  standard_deviations~ : @device.Allocation,
  means~ : @device.Allocation,
  max_bytes~ : Int64,
) -> Self raise @device.DeviceError {
  let layout = @joint_diffusion_device.VaeInputLayout::new(
    source.permutation,
    batch=1,
    channels=24,
    elements_per_channel=source.elements_per_channel,
    max_bytes=max_bytes / 2L,
  )
  guard means.byte_count() >= layout.parameter_bytes &&
    standard_deviations.byte_count() >= layout.parameter_bytes else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  {
    source,
    layout,
    storage: context.allocate(layout.value_bytes * 2L),
    permutation,
    destandardize,
    standard_deviations,
    means,
    closed: false,
  }
}

///|
/// Only list preparation: the destination is not ready until these launches
/// complete. Append the VideoVAE decode list after them on the same stream.
pub fn VideoInputProgram::bind(
  self : Self,
  input : @device.Allocation,
  input_offset : Int64,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard !self.closed else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  self.layout.launches(
    self.permutation,
    self.destandardize,
    input~,
    input_offset~,
    scratch=self.storage,
    scratch_offset=0L,
    output=self.storage,
    output_offset=self.layout.value_bytes,
    standard_deviations=self.standard_deviations,
    standard_deviations_offset=0L,
    means=self.means,
    means_offset=0L,
  )
}

///|
/// Borrow the destination for binding downstream work, not reading completed
/// output. The consuming queue must outlive neither this owner nor its producers.
pub fn[T] VideoInputProgram::with_destination(
  self : Self,
  consume : (@device.Allocation, Int64) -> T raise @device.DeviceError,
) -> T raise @device.DeviceError {
  guard !self.closed else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  consume(self.storage, self.layout.value_bytes)
}

///|
pub fn VideoInputProgram::matches_plan(
  self : Self,
  plan : @joint_diffusion_loop.PreparedPlan,
) -> Bool {
  !self.closed &&
  self.source.plan.model_digest() == plan.model_digest() &&
  self.source.plan.request() == plan.request()
}

///|
pub fn VideoInputProgram::close(self : Self) -> Unit raise @device.DeviceError {
  self.closed = true
  self.storage.close()
}
