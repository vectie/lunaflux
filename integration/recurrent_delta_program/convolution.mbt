///|
/// Compose the same Q/K/V convolution with a normalized recurrent update.
/// All three streams use the same geometry; asymmetric Q/K versus V kernels
/// remain available through the standalone recurrent constructor.
pub fn RecurrentDeltaProgram::with_convolution(
  source : @source.RecurrentDeltaCudaSource,
  convolution : @convolution.CausalConvolutionCudaSource,
) -> Self raise @precision.PrecisionPlanError {
  let (heads, keys, values) = source.plan().geometry()
  let (channels, _) = convolution.plan().geometry()
  guard keys == values &&
    heads.to_int64() * keys.to_int64() == channels.to_int64() &&
    source.plan().capacity() == convolution.plan().capacity() else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let state = convolution.plan().state_bytes()
  let frame = convolution.plan().frame_bytes()
  let delta = source.plan().state_bytes()
  guard state <= (9223372036854775807L - delta) / 3L else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  let cache = delta + 3L * state
  guard frame <= (9223372036854775807L - cache) / 3L else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  let (maximum_rows, _, _) = source.plan().capacity()
  {
    source,
    convolution: Some(convolution),
    block: None,
    maximum_rows,
    cache_bytes: cache,
    owned_bytes: cache + 3L * frame,
    launch_count: 4,
    status: Fresh,
    cache: None,
    functions: [],
    intermediates: [],
    queue: None,
  }
}

///|
fn initialize_cache(
  context : @device.Context,
  allocation : @device.Allocation,
  bytes : Int64,
) -> Unit raise @device.DeviceError {
  let zero = FixedArray::make(65536, b'\x00')
  for offset = 0L; offset < bytes; {
    let remaining = bytes - offset
    let count = if remaining < 65536L { remaining } else { 65536L }
    context.copy_from_fixed_host(allocation, zero, 0L, offset, count)
    continue offset + count
  }
}

///|
fn region(
  allocation : @device.Allocation,
  bytes : Int64,
  alignment : Int64,
) -> @device.KernelArgument raise @device.DeviceError {
  @device.KernelArgument::device_region(
    allocation,
    offset=0L,
    byte_count=bytes,
    alignment~,
  )
}

///|
fn convolution_launch(
  source : @convolution.CausalConvolutionCudaSource,
  function : @device.Function,
  counts : @device.Allocation,
  offsets : @device.Allocation,
  slots : @device.Allocation,
  reset : @device.Allocation,
  input : @device.Allocation,
  weight : @device.Allocation,
  state : @device.Allocation,
  output : @device.Allocation,
) -> @device.OrderedKernelLaunch raise @device.DeviceError {
  let plan = source.plan()
  let (channels, kernel) = plan.geometry()
  let (_, sequences, _) = plan.capacity()
  let (grid_x, grid_y, block_x) = source.launch()
  @device.OrderedKernelLaunch::new(
    function,
    @device.KernelLaunchDimensions::new(
      grid_x~,
      grid_y~,
      grid_z=1,
      block_x~,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    @device.KernelArguments::new([
      region(counts, 20L, 4L),
      region(offsets, (sequences.to_int64() + 1L) * 4L, 4L),
      region(slots, sequences.to_int64() * 4L, 4L),
      region(reset, sequences.to_int64() * 4L, 4L),
      region(input, plan.frame_bytes(), 2L),
      region(weight, channels.to_int64() * kernel.to_int64() * 2L, 2L),
      region(state, plan.state_bytes(), 2L),
      region(output, plan.frame_bytes(), 2L),
    ]),
  )
}
