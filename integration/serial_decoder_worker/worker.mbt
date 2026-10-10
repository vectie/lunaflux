///|
pub(all) enum DecoderInputPort {
  Counts
  TokenIds
  SequenceOffsets
  SequenceSlots
  RecurrentResets
  SparseReset
  OutputCount
  SelectedRows
  Positions
} derive(Debug, Eq)

///|
pub extend DecoderInputPort with @debug.Debug::{to_repr}

///|
pub extend DecoderInputPort with Eq::{equal, not_equal}

///|
fn DecoderInputPort::index(self : Self) -> Int {
  match self {
    Counts => 0
    TokenIds => 1
    SequenceOffsets => 2
    SequenceSlots => 3
    RecurrentResets => 4
    SparseReset => 5
    OutputCount => 6
    SelectedRows => 7
    Positions => 8
  }
}

///|
priv enum WorkerState {
  Fresh
  PortsPrepared
  Ready
  Uploading
  InputsReady
  Executing
  Downloading
  Retired
  Failed
  Closed
}

///|
priv struct DecoderEffects {
  submit : (Int) -> Unit raise
  poll : () -> Bool raise
  close : () -> Unit raise
}

///|
/// A prepared, model-neutral worker bridge. Startup binds the decoder effects;
/// the warmed path only copies request payloads, polls events and publishes the
/// existing completion protocol. No TLS, artifact scans or model switches.
/// Decoder effects own their queue and drain it when close() is called.
pub struct SerialDecoderWorker {
  priv frame : @frame.SerialDecoderFrame
  priv inputs : Array[@device.Allocation]
  priv pools : Array[@device.TransferPool]
  priv payloads : Array[FixedArray[Byte]]
  priv mut sampled : @device.Allocation?
  priv mut effects : DecoderEffects?
  priv whole_model : Bool
  priv outputs_token : Bool
  priv vocabulary : Int
  priv mut stage_failed : Bool
  priv mut stage_token : Int
  priv mut state : WorkerState
}

///|
pub fn SerialDecoderWorker::new(frame : @frame.SerialDecoderFrame) -> Self {
  let (_, _, vocabulary) = frame.geometry()
  {
    frame,
    inputs: [],
    pools: [],
    payloads: [],
    sampled: None,
    effects: None,
    whole_model: true,
    outputs_token: true,
    vocabulary,
    stage_failed: false,
    stage_token: 0,
    state: Fresh,
  }
}

///|
pub fn SerialDecoderWorker::geometry(self : Self) -> (Int, Int, Int) {
  self.frame.geometry()
}

///|
pub fn SerialDecoderWorker::output_capacity(self : Self) -> Int {
  self.frame.output_capacity()
}

///|
/// Counts, metadata, live token capacity and one selected token. Decoder
/// residuals, weights, state and AOT module are budgeted by their own owners.
pub fn SerialDecoderWorker::required_device_bytes(self : Self) -> Int64 raise {
  for
    port in 0..<self.frame.input_count()
    bytes = self.output_capacity().to_int64() * 4L {
    continue bytes + self.frame.input_capacity(port).to_int64()
  } nobreak {
    bytes
  }
}

///|
fn SerialDecoderWorker::prepare_pool(
  self : Self,
  context : @device.Context,
  allocation : @device.Allocation,
  bytes : Int,
) -> Unit raise {
  let pool = context.prepare_transfer_pool(
    allocation,
    host_bytes=bytes.to_int64(),
    lanes=1,
    max_regions_per_lane=1,
  )
  self.pools.push(pool)
  pool.check_prepared()
  self.payloads.push(FixedArray::make(bytes, b'\x00'))
}

///|
/// Acquire all transfer resources once, before binding the model queue to
/// these ports. A failed preparation remains owned and explicitly closeable.
pub fn SerialDecoderWorker::prepare_ports(
  self : Self,
  context : @device.Context,
  maximum_device_bytes~ : Int64,
) -> Unit raise {
  guard self.state is Fresh &&
    self.required_device_bytes() <= maximum_device_bytes else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.state = Failed
  for port in 0..<self.frame.input_count() {
    let bytes = self.frame.input_capacity(port)
    let allocation = context.allocate(bytes.to_int64())
    self.inputs.push(allocation)
    self.prepare_pool(context, allocation, bytes)
  }
  let sample_bytes = self.output_capacity() * 4
  let sampled = context.allocate(sample_bytes.to_int64())
  self.sampled = Some(sampled)
  self.prepare_pool(context, sampled, sample_bytes)
  self.state = PortsPrepared
}

///|
/// Borrowed until close; intended for startup binding, not token dispatch.
pub fn SerialDecoderWorker::input(
  self : Self,
  port : DecoderInputPort,
) -> @device.Allocation raise {
  guard self.state is PortsPrepared || self.state is Ready else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  guard port.index() < self.inputs.length() else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.inputs[port.index()]
}

///|
pub fn SerialDecoderWorker::sampled_tokens(
  self : Self,
) -> @device.Allocation raise {
  guard (self.state is PortsPrepared || self.state is Ready) &&
    self.sampled is Some(sampled) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  sampled
}

///|
/// Append descriptors are borrowed [base, live, error] I32 ports. Their error
/// word is read only after queue completion, as part of output delivery. This
/// is not a qualification-only diagnostic round trip.
pub fn SerialDecoderWorker::bind_executor(
  self : Self,
  context : @device.Context,
  submit~ : (Int) -> Unit raise,
  poll~ : () -> Bool raise,
  close~ : () -> Unit raise,
  append_descriptors~ : ArrayView[@device.Allocation],
) -> Unit raise {
  guard self.whole_model && self.output_capacity() == 1 else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.bind_effects(context, submit~, poll~, close~, append_descriptors~)
}

///|
fn SerialDecoderWorker::bind_effects(
  self : Self,
  context : @device.Context,
  submit~ : (Int) -> Unit raise,
  poll~ : () -> Bool raise,
  close~ : () -> Unit raise,
  append_descriptors~ : ArrayView[@device.Allocation],
) -> Unit raise {
  guard self.state is PortsPrepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.state = Failed
  self.effects = Some({ submit, poll, close, })
  for descriptor in append_descriptors {
    guard descriptor.byte_count() >= 12L else {
      raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
    }
    self.prepare_pool(context, descriptor, 4)
  }
  self.state = Ready
}

///|
pub fn SerialDecoderWorker::begin(
  self : Self,
  frame : @wire.ValidatedPlanFrame,
  writer : @wire.CompletionFrameWriter,
) -> Unit raise {
  guard self.whole_model else { raise @frame.SerialDecoderError::Busy }
  self.begin_inputs(frame, writer)
}

///|
fn SerialDecoderWorker::begin_inputs(
  self : Self,
  frame : @wire.ValidatedPlanFrame,
  writer : @wire.CompletionFrameWriter,
) -> Unit raise {
  guard self.state is Ready else { raise @frame.SerialDecoderError::Busy }
  // Rejected plans have not entered DMA or changed retained model state.
  self.frame.stage(frame, writer)
  self.state = Failed
  for port in 0..<self.frame.input_count() {
    let bytes = self.frame.copy_input(port, self.payloads[port])
    let pool = self.pools[port]
    pool.write_host(
      self.payloads[port],
      host_offset=0L,
      source_offset=0,
      bytes~,
    )
    pool.enqueue(
      lane=0,
      host_offset=0L,
      device_offset=0L,
      bytes=bytes.to_int64(),
      to_device=true,
    )
    pool.record(0)
  }
  self.state = Uploading
}

///|
fn SerialDecoderWorker::uploads_done(self : Self) -> Bool raise {
  for port in 0..<self.frame.input_count() {
    if !self.pools[port].poll(0) {
      return false
    }
  }
  for port in 0..<self.frame.input_count() {
    self.pools[port].reset(0)
  }
  true
}

///|
fn SerialDecoderWorker::download(self : Self) -> Unit raise {
  let sample_port = self.frame.input_count()
  for port in sample_port..<self.pools.length() {
    if port != sample_port ||
      (self.outputs_token && self.frame.produces_token()) {
      let pool = self.pools[port]
      pool.enqueue(
        lane=0,
        host_offset=0L,
        device_offset=if port == sample_port { 0L } else { 8L },
        bytes=if port == sample_port {
          self.frame.output_rows().to_int64() * 4L
        } else {
          4L
        },
        to_device=false,
      )
      pool.record(0)
    }
  }
}

///|
fn SerialDecoderWorker::downloads_done(self : Self) -> Bool raise {
  let sample_port = self.frame.input_count()
  for port in sample_port..<self.pools.length() {
    if (
        port != sample_port ||
        (self.outputs_token && self.frame.produces_token())
      ) &&
      !self.pools[port].poll(0) {
      return false
    }
  }
  for port in sample_port..<self.pools.length() {
    if port != sample_port ||
      (self.outputs_token && self.frame.produces_token()) {
      self.pools[port].read_host(
        self.payloads[port],
        host_offset=0L,
        destination_offset=0,
        bytes=if port == sample_port { self.frame.output_rows() * 4 } else { 4 },
      )
      self.pools[port].reset(0)
    }
  }
  true
}

///|
fn load_i32(bytes : FixedArray[Byte], row? : Int = 0) -> Int {
  for i in 0..<4; bits = 0U {
    continue bits | (bytes[row * 4 + i].to_uint() << (i * 8))
  } nobreak {
    bits.reinterpret_as_int()
  }
}

///|
/// One nonblocking progress phase. Returns true when the canonical completion,
/// exact model queue and output transfers retire. The caller keeps the plan
/// frame unchanged until then and transports the result using existing wire IO.
pub fn SerialDecoderWorker::progress(self : Self) -> Bool raise {
  guard self.whole_model else { raise @frame.SerialDecoderError::Busy }
  guard self.effects is Some(effects) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  match self.state {
    Uploading => {
      self.state = Failed
      if !self.uploads_done() {
        self.state = Uploading
        return false
      }
      (effects.submit)(self.frame.live_rows())
      self.state = Executing
      false
    }
    Executing => {
      self.state = Failed
      if !(effects.poll)() {
        self.state = Executing
        return false
      }
      self.download()
      self.state = Downloading
      false
    }
    Downloading => {
      self.state = Failed
      if !self.downloads_done() {
        self.state = Downloading
        return false
      }
      let sample_port = self.frame.input_count()
      let failed = self.payloads[sample_port + 1:].any(bytes => {
        load_i32(bytes) != 0
      })
      let token = load_i32(self.payloads[sample_port])
      ignore(self.frame.complete(sampled_token=token, execution_failed=failed))
      // Frame completion validates both lower and upper vocabulary bounds.
      self.state = if self.frame.is_faulted() { Failed } else { Ready }
      true
    }
    _ => raise @frame.SerialDecoderError::Busy
  }
}

///|
/// Read the published wire value without boxing a polling result.
pub fn SerialDecoderWorker::completion(
  self : Self,
) -> @wire.ValidatedCompletionFrame raise {
  guard self.whole_model && (self.state is Ready || self.state is Failed) else {
    raise @frame.SerialDecoderError::Busy
  }
  self.frame.completion()
}

///|
pub fn SerialDecoderWorker::release_request(self : Self) -> Unit raise {
  guard self.state is Ready else { raise @frame.SerialDecoderError::Busy }
  self.frame.release_request()
}

///|
/// Retire DMA and drop its allocation leases, then stop the borrowing decoder
/// queue before releasing request ports. The decoder may own append descriptors
/// and close them with its layers; no transfer lease can outlive that close.
/// Normal progress never calls drain or a blocking device synchronization.
pub fn SerialDecoderWorker::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  while self.pools.last() is Some(pool) {
    pool.drain(0)
    pool.close()
    ignore(self.pools.pop())
  }
  if self.effects is Some(effects) {
    (effects.close)()
    self.effects = None
  }
  self.frame.close()
  if self.sampled is Some(sampled) {
    sampled.close()
    self.sampled = None
  }
  while self.inputs.last() is Some(input) {
    input.close()
    ignore(self.inputs.pop())
  }
  self.payloads.clear()
  self.state = Closed
}
