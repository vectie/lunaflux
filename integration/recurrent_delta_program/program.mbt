///|
priv enum State {
  Fresh
  Ready
  Active
  Failed
  Closed
}

///|
/// Owns one layer's recurrent cache and AOT queue. All frame operands and the
/// stream/module are borrowed. Request-slot IDs survive batch reordering;
/// resetting a slot begins a new request, not a new batch.
pub struct RecurrentDeltaProgram {
  priv source : @source.RecurrentDeltaCudaSource
  priv convolution : @convolution.CausalConvolutionCudaSource?
  priv block : @block.RecurrentBlockCudaSource?
  priv maximum_rows : Int
  priv cache_bytes : Int64
  priv owned_bytes : Int64
  priv mut launch_count : Int
  priv mut status : State
  priv mut cache : @device.Allocation?
  priv functions : Array[@device.Function]
  priv intermediates : Array[@device.Allocation]
  priv mut queue : @device.OrderedKernelExecutor?
}

///|
pub fn RecurrentDeltaProgram::new(
  source : @source.RecurrentDeltaCudaSource,
) -> Self {
  let (maximum_rows, _, _) = source.plan().capacity()
  {
    source,
    convolution: None,
    block: None,
    maximum_rows,
    cache_bytes: source.plan().state_bytes(),
    owned_bytes: source.plan().state_bytes(),
    launch_count: 1,
    status: Fresh,
    cache: None,
    functions: [],
    intermediates: [],
    queue: None,
  }
}

///|
pub fn RecurrentDeltaProgram::state_bytes(self : Self) -> Int64 {
  self.cache_bytes
}

///|
pub fn RecurrentDeltaProgram::workspace_bytes(self : Self) -> Int64 {
  self.owned_bytes
}

///|
pub fn RecurrentDeltaProgram::source_bytes(self : Self) -> Bytes {
  if self.block is Some(block) {
    return block.source_bytes()
  }
  match self.convolution {
    None => self.source.source_bytes()
    Some(convolution) =>
      convolution.source_bytes() + b"\n" + self.source.source_bytes()
  }
}

///|
/// Prepare once. CSR offsets cover live rows; slots are exclusive within each
/// frame and reset flags mark new requests. Metadata may be updated in-place
/// only while idle. State initialization uses bounded startup scratch.
pub fn RecurrentDeltaProgram::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  stream : @device.Stream,
  maximum_state_bytes~ : Int64,
  maximum_workspace_bytes? : Int64 = maximum_state_bytes,
  counts~ : @device.Allocation,
  sequence_offsets~ : @device.Allocation,
  sequence_slots~ : @device.Allocation,
  reset_flags~ : @device.Allocation,
  query~ : @device.Allocation,
  key~ : @device.Allocation,
  value~ : @device.Allocation,
  log_decay~ : @device.Allocation,
  beta~ : @device.Allocation,
  output~ : @device.Allocation,
  convolution_weights? : FixedArray[@device.Allocation],
  block_weights? : RecurrentBlockWeights,
  hidden? : @device.Allocation,
  prefix? : ArrayView[@device.OrderedKernelLaunch] = [],
  suffix? : ArrayView[@device.OrderedKernelLaunch] = [],
) -> Unit raise @device.DeviceError {
  guard self.status is Fresh &&
    self.cache is None &&
    self.functions.is_empty() &&
    self.intermediates.is_empty() &&
    self.state_bytes() <= maximum_state_bytes &&
    self.workspace_bytes() <= maximum_workspace_bytes else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let plan = self.source.plan()
  let (heads, keys, values) = plan.geometry()
  let (rows, sequences, _) = plan.capacity()
  let qbytes = rows.to_int64() * heads.to_int64() * keys.to_int64() * 2L
  let vbytes = rows.to_int64() * heads.to_int64() * values.to_int64() * 2L
  // Check startup binding shape before acquiring any owned buffer.
  match (self.convolution, convolution_weights) {
    (None, None) => ()
    (Some(_), Some(weights)) if weights.length() == 3 => ()
    _ => raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  match (self.block, block_weights, hidden) {
    (None, None, None) => ()
    (Some(_), Some(_), Some(_)) => ()
    _ => raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let state = context.allocate(plan.state_bytes())
  self.cache = Some(state)
  initialize_cache(context, state, plan.state_bytes())
  // Startup composition only. Borrowed envelope functions/regions must outlive
  // this one queue; token submission never constructs or merges launch arrays.
  let launches = prefix.to_array()
  let frame = match (self.block, block_weights, hidden) {
    (Some(block), Some(weights), Some(hidden)) => {
      let frame = allocate_block_frame(self, context, block)
      append_block_prefix(
        self, aot, block, launches, counts, hidden, weights, frame,
      )
      Some(frame)
    }
    _ => None
  }
  let inputs = match frame {
    Some(frame) => [frame[0], frame[1], frame[2]]
    None => [query, key, value]
  }
  // Keep convolution-owned buffer offsets independent of the block prefix.
  let convolution_start = self.intermediates.length()
  let delta_inputs = match (self.convolution, convolution_weights) {
    (Some(convolution), Some(weights)) => {
      for i in 0..<3 {
        let history = context.allocate(convolution.plan().state_bytes())
        self.intermediates.push(history)
        initialize_cache(context, history, convolution.plan().state_bytes())
        let convolved = context.allocate(convolution.plan().frame_bytes())
        self.intermediates.push(convolved)
        let function = aot.load_function(@utf8.encode(convolution.symbol()))
        self.functions.push(function)
        launches.push(
          convolution_launch(
            convolution,
            function,
            counts,
            sequence_offsets,
            sequence_slots,
            reset_flags,
            inputs[i],
            weights[i],
            history,
            convolved,
          ),
        )
      }
      [
        self.intermediates[convolution_start + 1],
        self.intermediates[convolution_start + 3],
        self.intermediates[convolution_start + 5],
      ]
    }
    _ => inputs
  }
  let function = aot.load_function(@utf8.encode(self.source.symbol()))
  self.functions.push(function)
  let (grid_x, grid_y, block_x) = self.source.launch()
  let launch = @device.OrderedKernelLaunch::new(
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
      region(sequence_offsets, (sequences.to_int64() + 1L) * 4L, 4L),
      region(sequence_slots, sequences.to_int64() * 4L, 4L),
      region(reset_flags, sequences.to_int64() * 4L, 4L),
      region(delta_inputs[0], qbytes, 2L),
      region(delta_inputs[1], qbytes, 2L),
      region(delta_inputs[2], vbytes, 2L),
      region(
        match frame {
          Some(frame) => frame[7]
          None => log_decay
        },
        qbytes * 2L,
        4L,
      ),
      region(
        match frame {
          Some(frame) => frame[6]
          None => beta
        },
        rows.to_int64() * heads.to_int64() * 2L,
        2L,
      ),
      region(state, plan.state_bytes(), 4L),
      region(
        match frame {
          Some(frame) => frame[10]
          None => output
        },
        vbytes,
        2L,
      ),
    ]),
  )
  launches.push(launch)
  match (self.block, block_weights, frame) {
    (Some(block), Some(weights), Some(frame)) =>
      append_block_suffix(
        self, aot, block, launches, counts, weights, frame, output,
      )
    _ => ()
  }
  for launch in suffix {
    launches.push(launch)
  }
  self.queue = Some(context.create_ordered_kernel_executor(stream, launches))
  self.launch_count = launches.length()
  self.status = Ready
}

///|
pub fn RecurrentDeltaProgram::submit(
  self : Self,
  live_rows~ : Int,
) -> Unit raise @device.DeviceError {
  guard self.status is Ready && self.queue is Some(queue) else {
    raise @device.DeviceError::Busy(OrderedExecutorEnqueue)
  }
  guard live_rows > 0 && live_rows <= self.maximum_rows else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  errdefer {
    self.status = Failed
  }
  for index in 0..<self.launch_count {
    queue.enqueue(index)
  }
  queue.record_completion()
  self.status = Active
}

///|
pub fn RecurrentDeltaProgram::poll(
  self : Self,
) -> Bool raise @device.DeviceError {
  guard self.status is Active && self.queue is Some(queue) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  errdefer {
    self.status = Failed
  }
  if queue.poll_completion() {
    queue.reset()
    self.status = Ready
    true
  } else {
    false
  }
}

///|
pub fn RecurrentDeltaProgram::close(
  self : Self,
) -> Unit raise @device.DeviceError {
  if self.status is Closed {
    return
  }
  guard self.status is (Fresh | Ready) else {
    raise @device.DeviceError::Busy(OrderedExecutorClose)
  }
  if self.queue is Some(queue) {
    queue.close()
    self.queue = None
  }
  while self.functions.last() is Some(function) {
    function.close()
    ignore(self.functions.pop())
  }
  while self.intermediates.last() is Some(allocation) {
    allocation.close()
    ignore(self.intermediates.pop())
  }
  if self.cache is Some(cache) {
    cache.close()
    self.cache = None
  }
  self.status = Closed
}

///|
/// Cancellation drains before releasing state. Partially updated request
/// state is never reused after abort: this owner becomes terminal.
pub fn RecurrentDeltaProgram::abort(
  self : Self,
) -> Unit raise @device.DeviceError {
  if self.status is Closed {
    return
  }
  if self.queue is Some(queue) {
    self.status = Failed
    queue.abort()
  }
  self.status = Fresh
  self.close()
}
