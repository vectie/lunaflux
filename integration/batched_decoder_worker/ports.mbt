///|
priv enum State {
  Fresh
  Prepared
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
priv struct Effects {
  submit : (Int) -> Unit raise
  poll : () -> Bool raise
  close : () -> Unit raise
}

///|
/// Rank-local effect owner. All arrays, DMA lanes and device addresses are
/// prepared once. Only the coordinator may commit after every rank retires.
/// A rank never publishes a successful model completion on its own.
pub struct BatchedDecoderRank {
  priv frame : @frame.BatchedDecoderFrame
  priv ports : ReadOnlyArray[@frame.BatchInputPort]
  priv inputs : Array[@device.Allocation]
  priv pools : Array[@device.TransferPool]
  priv payloads : Array[FixedArray[Byte]]
  priv live_bytes : FixedArray[Int]
  priv samples : Array[Int]
  priv outputs_token : Bool
  priv vocabulary : Int
  priv mut sampled : @device.Allocation?
  priv mut effects : Effects?
  priv mut failed : Bool
  priv mut state : State
}

///|
pub fn BatchedDecoderRank::new(
  frame : @frame.BatchedDecoderFrame,
  outputs_token~ : Bool,
) -> Self {
  let (_, sequences, _, _, vocabulary) = frame.geometry()
  {
    frame,
    ports: ReadOnlyArray::from_array([
      Counts,
      TokenIds,
      SequenceOffsets,
      SequenceSlots,
      RecurrentResets,
      SparseResets,
      OutputCount,
      SelectedRows,
      Positions,
      RowSlots,
      SequencePositions,
      RowSequences,
      PageOffsets,
      PageIds,
      PageGenerations,
    ]),
    inputs: [],
    pools: [],
    payloads: [],
    live_bytes: FixedArray::make(15, 0),
    samples: Array::make(sequences, 0),
    outputs_token,
    vocabulary,
    sampled: None,
    effects: None,
    failed: false,
    state: Fresh,
  }
}

///|
pub fn BatchedDecoderRank::geometry(self : Self) -> (Int, Int, Int, Int, Int) {
  self.frame.geometry()
}

///|
pub fn BatchedDecoderRank::outputs_token(self : Self) -> Bool {
  self.outputs_token
}

///|
pub fn BatchedDecoderRank::required_device_bytes(self : Self) -> Int64 {
  for port in self.ports; bytes = self.samples.length().to_int64() * 4L {
    continue bytes + Int::max(4, self.frame.input_capacity(port)).to_int64()
  } nobreak {
    bytes
  }
}

///|
fn BatchedDecoderRank::prepare_pool(
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
pub fn BatchedDecoderRank::prepare_ports(
  self : Self,
  context : @device.Context,
  maximum_device_bytes~ : Int64,
) -> Unit raise {
  guard self.state is Fresh &&
    self.required_device_bytes() <= maximum_device_bytes else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.state = Failed
  for port in self.ports {
    // Unused optional ports still have a valid startup address; they never DMA.
    let bytes = Int::max(4, self.frame.input_capacity(port))
    let input = context.allocate(bytes.to_int64())
    self.inputs.push(input)
    self.prepare_pool(context, input, bytes)
  }
  let sampled = context.allocate(self.samples.length().to_int64() * 4L)
  self.sampled = Some(sampled)
  self.prepare_pool(context, sampled, self.samples.length() * 4)
  self.state = Prepared
}

///|
pub fn BatchedDecoderRank::input(
  self : Self,
  port : @frame.BatchInputPort,
) -> @device.Allocation raise {
  guard self.state is Prepared || self.state is Ready else {
    raise @frame.BatchFrameError::Busy
  }
  for index, candidate in self.ports {
    if candidate == port {
      return self.inputs[index]
    }
  }
  raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
}

///|
pub fn BatchedDecoderRank::sampled_tokens(
  self : Self,
) -> @device.Allocation raise {
  guard (self.state is Prepared || self.state is Ready) &&
    self.sampled is Some(sampled) else {
    raise @frame.BatchFrameError::Busy
  }
  sampled
}

///|
pub fn BatchedDecoderRank::bind_executor(
  self : Self,
  context : @device.Context,
  submit~ : (Int) -> Unit raise,
  poll~ : () -> Bool raise,
  close~ : () -> Unit raise,
  append_descriptors~ : ArrayView[@device.Allocation],
) -> Unit raise {
  guard self.state is Prepared else { raise @frame.BatchFrameError::Busy }
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
/// Drain transfer leases before closing the queue that owns append ports.
/// Only cancellation/teardown drains; normal progress is nonblocking.
pub fn BatchedDecoderRank::close(self : Self) -> Unit raise {
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
