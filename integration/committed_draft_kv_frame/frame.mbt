///|
priv enum State {
  Fresh
  Failed
  Prepared
  Closed
}

///|
/// One persistent main ring, two startup-built phase launch tables. The parent
/// queues own completion and cancellation; drain them before closing this owner.
pub struct CommittedDraftKvFrame {
  priv prime : @source.CommittedDraftKvCudaSource
  priv predict : @source.CommittedDraftKvCudaSource
  priv committed : @source.CommittedDraftKvCudaSource?
  priv buffers : Array[@device.Allocation]
  priv functions : Array[@device.Function]
  priv prime_launches : Array[@device.OrderedKernelLaunch]
  priv predict_launches : Array[@device.OrderedKernelLaunch]
  priv committed_launches : Array[@device.OrderedKernelLaunch]
  priv mut state : State
}

///|
pub fn CommittedDraftKvFrame::new(
  prime : @source.CommittedDraftKvCudaSource,
  predict : @source.CommittedDraftKvCudaSource,
  committed? : @source.CommittedDraftKvCudaSource,
) -> Self raise {
  guard prime.phase() is Prime &&
    predict.phase() is Predict &&
    prime.plan().geometry() == predict.plan().geometry() else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  if committed is Some(source) {
    guard source.phase() is PredictCommitted &&
      source.plan().geometry() == prime.plan().geometry() else {
      raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
    }
  }
  {
    prime,
    predict,
    committed,
    buffers: [],
    functions: [],
    prime_launches: [],
    predict_launches: [],
    committed_launches: [],
    state: Fresh,
  }
}

///|
pub fn CommittedDraftKvFrame::source_bytes(self : Self) -> Bytes {
  self.prime.source_bytes() +
  b"\n" +
  self.predict.source_bytes() +
  (match self.committed {
    Some(source) => b"\n" + source.source_bytes()
    None => b""
  })
}

///|
pub fn CommittedDraftKvFrame::state_bytes(self : Self) -> Int64 {
  self.prime.plan().state_bytes()
}

///|
fn region(
  b : @device.Allocation,
  bytes : Int64,
  alignment : Int64,
) -> @device.KernelArgument raise {
  @device.KernelArgument::device_region(
    b,
    offset=0L,
    byte_count=bytes,
    alignment~,
  )
}

///|
pub fn CommittedDraftKvFrame::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  main_counts~ : @device.Allocation,
  draft_counts~ : @device.Allocation,
  reset_flag~ : @device.Allocation,
  start_position~ : @device.Allocation,
  main_kv~ : @device.Allocation,
  queries~ : @device.Allocation,
  draft_kv~ : @device.Allocation,
  attention_sink~ : @device.Allocation,
  maximum_state_bytes~ : Int64,
) -> Unit raise {
  guard self.state is Fresh &&
    self.prime.plan().state_bytes() <= maximum_state_bytes else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  // Separate count planes cannot alias: prediction must not reinterpret main
  // prefill counts as a draft block, or vice versa.
  guard !main_counts.same_storage(draft_counts) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let (heads, width, _, main_rows, draft_rows, _) = self.prime.plan().geometry()
  let spans = [
    (main_counts, 20L, 4L),
    (draft_counts, 20L, 4L),
    (reset_flag, 4L, 4L),
    (start_position, 4L, 4L),
    (main_kv, main_rows.to_int64() * width.to_int64() * 2L, 2L),
    (
      queries,
      draft_rows.to_int64() * heads.to_int64() * width.to_int64() * 2L,
      2L,
    ),
    (draft_kv, draft_rows.to_int64() * width.to_int64() * 2L, 2L),
    (attention_sink, heads.to_int64() * 4L, 4L),
  ]
  for (b, bytes, alignment) in spans {
    context.validate_allocation_region(
      b,
      offset=0L,
      byte_count=bytes,
      alignment~,
    )
  }
  let ports = spans.map(span => {
    let (b, bytes, alignment) = span
    region(b, bytes, alignment)
  })
  self.state = Failed
  let sizes = self.prime.plan().buffer_bytes()
  for size in sizes {
    self.buffers.push(context.allocate(size))
  }
  self.buffers[1].copy_from_host(
    b"\x00\x00\x00\x00\x00\x00\x00\x00", 0L, 0L, 8L,
  )
  let ring = region(self.buffers[0], sizes[0], 2L)
  let history = region(self.buffers[1], sizes[1], 4L)
  let descriptor = region(self.buffers[2], sizes[2], 4L)
  let output = region(self.buffers[3], sizes[3], 2L)
  let args = [
    [ports[0], ports[1], ports[2], ports[3], history, descriptor],
    [descriptor, ports[4], ring],
    [descriptor, history],
    [descriptor, ports[5], ports[6], ring, ports[7], output],
  ]
  for
    (source, launches) in [
      (self.prime, self.prime_launches),
      (self.predict, self.predict_launches),
    ] {
    for stage in 0..<source.stage_count() {
      let f = aot.load_function(@utf8.encode(source.symbol(stage)))
      self.functions.push(f)
      let (x, y, threads) = source.launch(stage)
      launches.push(
        @device.OrderedKernelLaunch::new(
          f,
          @device.KernelLaunchDimensions::new(
            grid_x=x,
            grid_y=y,
            grid_z=1,
            block_x=threads,
            block_y=1,
            block_z=1,
            shared_memory_bytes=0,
          ),
          @device.KernelArguments::new(args[stage]),
        ),
      )
    }
  }
  if self.committed is Some(source) {
    for stage in 0..<source.stage_count() {
      let f = aot.load_function(@utf8.encode(source.symbol(stage)))
      self.functions.push(f)
      let (x, y, threads) = source.launch(stage)
      self.committed_launches.push(
        @device.OrderedKernelLaunch::new(
          f,
          @device.KernelLaunchDimensions::new(
            grid_x=x,
            grid_y=y,
            grid_z=1,
            block_x=threads,
            block_y=1,
            block_z=1,
            shared_memory_bytes=0,
          ),
          @device.KernelArguments::new(args[if stage == 0 { 0 } else { 3 }]),
        ),
      )
    }
  }
  self.state = Prepared
}

///|
/// Read committed KV without copying main operands or advancing the frontier.
pub fn CommittedDraftKvFrame::prediction_from_committed_launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared && self.committed is Some(_) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.committed_launches[:]
}

///|
pub fn CommittedDraftKvFrame::prime_launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.prime_launches[:]
}

///|
pub fn CommittedDraftKvFrame::prediction_launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.predict_launches[:]
}

///|
pub fn CommittedDraftKvFrame::output(self : Self) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.buffers[3]
}

///|
pub fn CommittedDraftKvFrame::descriptor(
  self : Self,
) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.buffers[2]
}

///|
pub fn CommittedDraftKvFrame::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.prime_launches.clear()
  self.predict_launches.clear()
  self.committed_launches.clear()
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
