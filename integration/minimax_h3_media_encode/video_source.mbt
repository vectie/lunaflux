///|
pub struct VideoSource {
  plan : @audio.VideoEncoder
  kernels : @lower.Source
}

///|
pub fn VideoSource::new(
  frames~ : Int,
  height~ : Int,
  width~ : Int,
  image~ : Bool,
  max_tiles~ : Int,
  max_scratch_bytes~ : Int64,
) -> Self raise @plan.PlanError {
  let plan = @audio.VideoEncoder::new(
    frames~,
    height~,
    width~,
    image~,
    max_tiles~,
    max_scratch_bytes~,
  )
  { plan, kernels: @lower.render(plan.graph), }
}

///|
fn VideoSource::bind(
  self : Self,
  functions : ArrayView[@device.Function],
  weights : Map[String, @device.KernelArgument],
  input : @device.Allocation,
  input_offset : Int64,
  noise : @device.Allocation,
  noise_offset : Int64,
  statistics : @device.Allocation,
  scratch : @device.Allocation,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  let g = self.plan.graph
  guard functions.length() == g.nodes.length() else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let args : Array[@device.KernelArgument] = []
  for id, value in g.values {
    args.push(
      match value.origin {
        Temporary => region(scratch, g.offsets[id], value.elements)
        Weight(name) =>
          match weights.get(name) {
            Some(arg) => arg
            None => raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
          }
        Input =>
          if id == 0 {
            region(input, input_offset, value.elements)
          } else if id == self.plan.noise_value {
            region(noise, noise_offset, value.elements)
          } else if id == self.plan.means_value {
            region(statistics, 0L, 24L)
          } else if id == self.plan.deviations_value {
            region(statistics, 96L, 24L)
          } else {
            raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
          }
      },
    )
  }
  let launches : Array[@device.OrderedKernelLaunch] = []
  for i, node in g.nodes {
    let operands = node.inputs[:].map(id => args[id])
    // Zeros has no read operands; all five ignored ABI positions use its valid
    // destination rather than a fake null/zero-size argument.
    while operands.length() < 5 {
      operands.push(args[node.output])
    }
    operands.push(args[node.output])
    let (grid, threads) = self.kernels.grids[i]
    launches.push(
      @device.OrderedKernelLaunch::new(
        functions[i],
        @device.KernelLaunchDimensions::new(
          grid_x=grid,
          grid_y=1,
          grid_z=1,
          block_x=threads,
          block_y=1,
          block_z=1,
          shared_memory_bytes=0,
        ),
        @device.KernelArguments::new(operands),
      ),
    )
  }
  launches
}
