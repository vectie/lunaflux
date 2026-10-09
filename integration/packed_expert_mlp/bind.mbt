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
/// Prepare the real three-stage expert program once at startup. Functions are
/// loaded from the generated AOT module in source.symbols order. All buffers
/// and functions are borrowed, and must outlive the ordered executor.
///
/// Counts[3] is the live row count. Routing supplies distinct expert IDs per
/// row and F32 weights; expert_map is plan.expert_map. The output is rank-local
/// F32 (not a finalized global BF16 result), suitable for a subsequent reduce.
/// Input, activation, weighted and output must be separate allocations.
pub fn bind(
  source : @source.PackedExpertMlpCudaSource,
  functions : ArrayView[@device.Function],
  counts~ : @device.Allocation,
  input~ : @device.Allocation,
  selected_indices~ : @device.Allocation,
  selected_weights~ : @device.Allocation,
  expert_map~ : @device.Allocation,
  bank~ : @device.Allocation,
  activation~ : @device.Allocation,
  weighted~ : @device.Allocation,
  output~ : @device.Allocation,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard functions.length() == 3 else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let p = source.plan()
  let (rows, hidden, _, experts, selected) = p.geometry()
  let jobs = rows.to_int64() * selected.to_int64()
  let cells = rows.to_int64() * hidden.to_int64()
  let (activation_bytes, weighted_bytes) = p.workspace_bytes()
  let c = region(counts, 20L, 4L)
  let x = region(input, cells * 2L, 2L)
  let ix = region(selected_indices, jobs * 4L, 4L)
  let sw = region(selected_weights, jobs * 4L, 4L)
  let em = region(expert_map, experts.to_int64() * 4L, 4L)
  let w = region(bank, p.bank_bytes(), 16L)
  let a = region(activation, activation_bytes, 2L)
  let b = region(weighted, weighted_bytes, 2L)
  let y = region(output, cells * 4L, 4L)
  let arguments = [
    [c, x, ix, em, w, a],
    [c, a, ix, sw, em, w, b],
    [c, ix, b, y],
  ]
  let launches = []
  for i, launch in source.launches() {
    launches.push(
      @device.OrderedKernelLaunch::new(
        functions[i],
        @device.KernelLaunchDimensions::new(
          grid_x=launch.0,
          grid_y=1,
          grid_z=1,
          block_x=launch.1,
          block_y=1,
          block_z=1,
          shared_memory_bytes=0,
        ),
        @device.KernelArguments::new(arguments[i]),
      ),
    )
  }
  launches
}
