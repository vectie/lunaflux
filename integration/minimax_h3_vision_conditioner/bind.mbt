///|
fn region(
  a : @device.Allocation,
  offset : Int64,
  bytes : Int64,
) -> @device.KernelArgument raise @device.DeviceError {
  @device.KernelArgument::device_region(
    a,
    offset~,
    byte_count=bytes,
    alignment=2L,
  )
}

///|
fn grid(
  x : Int,
  y : Int,
  threads : Int,
) -> @device.KernelLaunchDimensions raise @device.DeviceError {
  @device.KernelLaunchDimensions::new(
    grid_x=x,
    grid_y=y,
    grid_z=1,
    block_x=threads,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
}

///|
/// Patches are BF16 [rows,3,2,16,16] in processor merge-group order. Each
/// checkpoint allocation is packed exactly according to its WeightGroup. All
/// operands/functions are borrowed until consuming queue closure. Workspace,
/// outputs and patch input must be disjoint immutable/checkpoint regions.
pub fn VisionSource::bind(
  self : Self,
  functions : ArrayView[@device.Function],
  frontend : @device.Allocation,
  blocks : ArrayView[@device.Allocation],
  mergers : ArrayView[@device.Allocation],
  patches~ : @device.Allocation,
  workspace~ : @device.Allocation,
  output~ : @device.Allocation,
) -> Array[@device.OrderedKernelLaunch] raise {
  guard functions.length() == 13 &&
    blocks.length() == 27 &&
    mergers.length() == 4 &&
    workspace.byte_count() >= self.plan.workspace_bytes &&
    output.byte_count() >= self.plan.output_bytes else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  guard !physical_equal(workspace, output) &&
    !physical_equal(workspace, patches) &&
    !physical_equal(output, patches) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  for allocation in blocks {
    guard !physical_equal(allocation, workspace) &&
      !physical_equal(allocation, output) else {
      raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
    }
  }
  for allocation in mergers {
    guard !physical_equal(allocation, workspace) &&
      !physical_equal(allocation, output) else {
      raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
    }
  }
  guard !physical_equal(frontend, workspace) &&
    !physical_equal(frontend, output) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let p = self.plan
  let args = p.workspace_regions().map(r => region(workspace, r.0, r.1))
  let state = args[0]
  let norm = args[1]
  let qkv = args[2]
  let rotary = args[3]
  let attention = args[4]
  let branch = args[5]
  let mlp = args[6]
  let merged = args[7]
  let launches : Array[@device.OrderedKernelLaunch] = []
  fn append(
    index : Int,
    dimensions : @device.KernelLaunchDimensions,
    operands : Array[@device.KernelArgument],
  ) -> Unit raise @device.DeviceError {
    launches.push(
      @device.OrderedKernelLaunch::new(
        functions[index],
        dimensions,
        @device.KernelArguments::new(operands),
      ),
    )
  }
  fn linear(
    rows : Int,
    width : Int,
  ) -> @device.KernelLaunchDimensions raise @device.DeviceError {
    grid(((rows.to_int64() * width.to_int64() + 255L) / 256L).to_int(), 1, 256)
  }
  fn weight_args(
    layout : @vision.WeightGroup,
    allocation : @device.Allocation,
  ) -> ReadOnlyArray[@device.KernelArgument] raise @device.DeviceError {
    layout.regions.map(r => region(allocation, r.offset, r.bytes))
  }
  fn merge(index : Int) -> Unit raise {
    let w = weight_args(@vision.merger_weights(index), mergers[index])
    append(
      if index == 0 {
        2
      } else {
        10
      },
      grid(if index == 0 { p.rows } else { p.merged_rows }, 1, 1),
      [state, w[0], w[1], norm],
    )
    append(11, linear(p.merged_rows, 4608), [norm, w[2], w[3], merged])
    append(12, linear(p.merged_rows, 5120), [
      merged,
      w[4],
      w[5],
      region(output, p.output_offset(index), p.output_bytes / 4L),
    ])
  }
  let front = weight_args(@vision.frontend_weights(), frontend)
  append(0, linear(p.rows, 1152), [
    region(patches, 0L, p.rows.to_int64() * 1536L * 2L),
    front[0],
    front[1],
    state,
  ])
  append(1, linear(p.rows, 1152), [state, front[2]])
  for i in 0..<27 {
    let w = weight_args(@vision.block_weights(i), blocks[i])
    append(2, grid(p.rows, 1, 1), [state, w[0], w[1], norm])
    append(3, linear(p.rows, 3456), [norm, w[2], w[3], qkv])
    append(4, linear(p.rows, 3456), [qkv, rotary])
    append(5, grid(p.rows, 16, 1), [rotary, attention])
    append(6, linear(p.rows, 1152), [attention, w[4], w[5], branch])
    append(7, linear(p.rows, 1152), [state, branch, state])
    append(2, grid(p.rows, 1, 1), [state, w[6], w[7], norm])
    append(8, linear(p.rows, 4304), [norm, w[8], w[9], mlp])
    append(9, linear(p.rows, 1152), [mlp, w[10], w[11], branch])
    append(7, linear(p.rows, 1152), [state, branch, state])
    if i == 8 {
      merge(1)
    } else if i == 16 {
      merge(2)
    } else if i == 24 {
      merge(3)
    }
  }
  merge(0)
  launches
}
