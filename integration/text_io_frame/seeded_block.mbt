///|
/// Prepared seed/noise metadata. The parent owns completion and callers borrow
/// counts, IDs and positions; no host construction occurs on prediction steps.
pub struct SeededTokenBlockFrame {
  priv plan : @precision.SeededTokenBlockPrecision
  priv source : Bytes
  priv symbol : String
  priv buffers : Array[@device.Allocation]
  priv functions : Array[@device.Function]
  priv table : Array[@device.OrderedKernelLaunch]
  priv mut state : State
}

///|
pub fn SeededTokenBlockFrame::new(
  plan : @precision.SeededTokenBlockPrecision,
  symbol_prefix~ : String,
) -> Self raise {
  {
    plan,
    source: @source.seeded_token_block_source(plan, symbol_prefix~),
    symbol: symbol_prefix + "_seed",
    buffers: [],
    functions: [],
    table: [],
    state: Fresh,
  }
}

///|
pub fn SeededTokenBlockFrame::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn SeededTokenBlockFrame::workspace_bytes(self : Self) -> Int64 {
  self.plan.workspace_bytes()
}

///|
pub fn SeededTokenBlockFrame::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  main_counts~ : @device.Allocation,
  start_position~ : @device.Allocation,
  seed_token~ : @device.Allocation,
  output_chain~ : @device.Allocation,
  maximum_workspace_bytes~ : Int64,
) -> Unit raise {
  guard self.state is Fresh && self.workspace_bytes() <= maximum_workspace_bytes else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  let (_, rows, _, _) = self.plan.geometry()
  for
    (b, bytes) in [
      (main_counts, 20L),
      (start_position, 4L),
      (seed_token, 4L),
      (output_chain, (rows.to_int64() + 1L) * 4L),
    ] {
    context.validate_allocation_region(
      b,
      offset=0L,
      byte_count=bytes,
      alignment=4L,
    )
  }
  for read_port in [main_counts, start_position, seed_token] {
    guard !output_chain.same_storage(read_port) else {
      raise @precision.PrecisionPlanError::IllegalPlacement
    }
  }
  self.state = Failed
  for size in [20L, rows.to_int64() * 4L, rows.to_int64() * 4L] {
    self.buffers.push(context.allocate(size))
  }
  let f = aot.load_function(@utf8.encode(self.symbol))
  self.functions.push(f)
  self.table.push(
    @device.OrderedKernelLaunch::new(
      f,
      @device.KernelLaunchDimensions::new(
        grid_x=1,
        grid_y=1,
        grid_z=1,
        block_x=1,
        block_y=1,
        block_z=1,
        shared_memory_bytes=0,
      ),
      @device.KernelArguments::new([
        region(main_counts, 20L, 4L),
        region(start_position, 4L, 4L),
        region(seed_token, 4L, 4L),
        region(self.buffers[0], 20L, 4L),
        region(self.buffers[1], rows.to_int64() * 4L, 4L),
        region(self.buffers[2], rows.to_int64() * 4L, 4L),
        region(output_chain, (rows.to_int64() + 1L) * 4L, 4L),
      ]),
    ),
  )
  self.state = Prepared
}

///|
pub fn SeededTokenBlockFrame::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.table[:]
}

///|
pub fn SeededTokenBlockFrame::counts(self : Self) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.buffers[0]
}

///|
pub fn SeededTokenBlockFrame::token_ids(
  self : Self,
) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.buffers[1]
}

///|
pub fn SeededTokenBlockFrame::positions(
  self : Self,
) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.buffers[2]
}

///|
pub fn SeededTokenBlockFrame::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.table.clear()
  while self.functions.last() is Some(f) {
    f.close()
    ignore(self.functions.pop())
  }
  while self.buffers.last() is Some(b) {
    b.close()
    ignore(self.buffers.pop())
  }
  self.state = Closed
}
