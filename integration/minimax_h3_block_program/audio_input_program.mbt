///|
/// Request-owned bridge from denoiser packed stereo rows to VAE input.
/// Two disjoint regions retain permutation and destandardized values. Functions
/// and F32 statistics are borrowed; close consuming requests before this owner.
pub struct AudioInputProgram {
  priv layout : @joint_diffusion_device.VaeInputLayout
  priv storage : @device.Allocation
  priv permutation : @device.Function
  priv destandardize : @device.Function
  priv standard_deviations : @device.Allocation
  priv means : @device.Allocation
  priv mut closed : Bool
  priv lease : AudioDecoderLease
}

///|
/// Functions must match packed [2,time,32] -> [2,32,time], block size 256,
/// followed by the F32 channel-major destandardization ABI. Statistics begin
/// at zero and contain 32 entries each. This is preparation, not device JIT.
pub fn AudioInputProgram::allocate(
  context : @device.Context,
  permutation : @device.Function,
  destandardize : @device.Function,
  time~ : Int,
  standard_deviations~ : @device.Allocation,
  means~ : @device.Allocation,
  max_bytes~ : Int64,
) -> Self raise {
  let plan = @dense_permutation_plan.Permutation::new(
    [2, time, 32],
    [0, 2, 1],
    max_elements=max_bytes / 8L,
  )
  let layout = @joint_diffusion_device.VaeInputLayout::new(
    plan,
    batch=2,
    channels=32,
    elements_per_channel=time.to_int64(),
    max_bytes=max_bytes / 2L,
  )
  guard standard_deviations.byte_count() >= layout.parameter_bytes &&
    means.byte_count() >= layout.parameter_bytes else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  {
    layout,
    storage: context.allocate(layout.value_bytes * 2L),
    permutation,
    destandardize,
    standard_deviations,
    means,
    closed: false,
    lease: { held: false, },
  }
}

///|
fn AudioInputProgram::bind(
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
pub fn AudioInputProgram::close(self : Self) -> Unit raise @device.DeviceError {
  self.lease.require_idle()
  self.closed = true
  self.storage.close()
}
