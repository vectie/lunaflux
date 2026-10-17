///|
/// Startup-only byte permutation. The caller lends the bank exclusively until
/// completion; its selected source defines the physical representation. One
/// expert scratch is reused across the bank, released before any token graph.
fn MoeProgram::prepare_bank(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  source : @source.PackedExpertMlpCudaSource,
  bank : @device.Allocation,
) -> Unit raise @device.DeviceError {
  guard source.bank_preparation() is Some(symbol) else { return }
  let stride = source.plan().expert_stride()
  let scratch = self.allocate(context, stride)
  let stream = context.create_stream()
  self.startup_streams.push(stream)
  let functions = self.load_functions(aot, [symbol, symbol + "_copy"])
  let shape = @device.KernelLaunchDimensions::new(
    grid_x=256,
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  let temporary = region(scratch, stride, 16L)
  for expert in 0..<source.plan().local_experts().length() {
    let slice = @device.KernelArgument::device_region(
      bank,
      offset=expert.to_int64() * stride,
      byte_count=stride,
      alignment=16L,
    )
    functions[0].launch_synchronous(
      shape,
      @device.KernelArguments::new([slice, temporary]),
      stream,
    )
    functions[1].launch_synchronous(
      shape,
      @device.KernelArguments::new([temporary, slice]),
      stream,
    )
  }
  stream.close()
  ignore(self.startup_streams.pop())
  // Scratch remains owned if any earlier operation throws; close releases it.
  scratch.close()
  ignore(self.allocations.pop())
}
