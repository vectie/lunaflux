///|
priv enum FrameState {
  Fresh
  Failed
  Prepared
  Closed
}

///|
/// Owns one scatter function and two metadata allocations; borrows text
/// launches and vision feature storage. Close request queues before this frame.
pub struct Frame {
  priv program : Program
  priv functions : Array[@device.Function]
  priv allocations : Array[@device.Allocation]
  priv prefill : Array[@device.OrderedKernelLaunch]
  priv decode : Array[@device.OrderedKernelLaunch]
  priv host_map : FixedArray[Byte]
  priv host_count : FixedArray[Byte]
  priv mut features : @device.Allocation?
  priv mut feature_capacity : Int32
  priv mut ready : Bool
  priv mut state : FrameState
}

///|
pub fn Frame::new(program : Program) -> Self {
  let (_, _, positions, _) = program.scatter.plan().geometry()
  {
    program,
    functions: [],
    allocations: [],
    prefill: [],
    decode: [],
    host_map: FixedArray::make(positions * 4, b'\xff'),
    host_count: FixedArray::make(4, b'\x00'),
    features: None,
    feature_capacity: 0,
    ready: false,
    state: Fresh,
  }
}

///|
pub fn Frame::prepare(
  self : Self,
  context : @device.Context,
  aot~ : @device.Module,
  text_frame~ : @text.Frame,
  text_storage~ : @text.Storage,
  features~ : @device.Allocation,
  maximum_metadata_bytes~ : Int64,
) -> Unit raise {
  guard self.state is Fresh &&
    maximum_metadata_bytes >= self.program.metadata_bytes() else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  let (width, rows, _, maximum_features) = self.program.scatter
    .plan()
    .geometry()
  let row_bytes = width.to_int64() * 2L
  let capacity = features.byte_count() / row_bytes
  guard features.byte_count() % row_bytes == 0L && capacity > 0L else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let prefill = text_frame.launches(Prefill)
  let decode = text_frame.launches(Decode)
  guard prefill.length() == self.program.text.launch_count(Prefill) &&
    decode.length() == self.program.text.launch_count(Decode) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let metadata = text_storage.metadata().decoder.attention
  let hidden = text_storage.buffers().hidden
  guard !features.same_storage(hidden) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let regions = [
    (metadata.counts, 20L, 4L),
    (metadata.causal_positions, rows.to_int64() * 4L, 4L),
    (features, capacity.min(maximum_features.to_int64()) * row_bytes, 2L),
    (hidden, rows.to_int64() * row_bytes, 2L),
  ]
  for (allocation, bytes, alignment) in regions {
    context.validate_allocation_region(
      allocation,
      offset=0L,
      byte_count=bytes,
      alignment~,
    )
  }
  self.state = Failed
  let map = context.allocate(self.program.metadata_bytes() - 4L)
  self.allocations.push(map)
  let count = context.allocate(4L)
  self.allocations.push(count)
  context.copy_from_fixed_host(count, self.host_count, 0L, 0L, 4L)
  let function = aot.load_function(@utf8.encode(self.program.scatter.symbol()))
  self.functions.push(function)
  let arguments = [
    metadata.counts,
    metadata.causal_positions,
    map,
    count,
    features,
    hidden,
  ]
  let sizes = [
    20L,
    rows.to_int64() * 4L,
    self.program.metadata_bytes() - 4L,
    4L,
    capacity.min(maximum_features.to_int64()) * row_bytes,
    rows.to_int64() * row_bytes,
  ]
  let parameters = arguments.mapi((i, allocation) => {
    @device.KernelArgument::device_region(
      allocation,
      offset=0L,
      byte_count=sizes[i],
      alignment=if i < 4 { 4L } else { 2L },
    )
  })
  let (grid_x, grid_y, block_x) = self.program.scatter.launch()
  let scatter = @device.OrderedKernelLaunch::new(
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
    @device.KernelArguments::new(parameters[:]),
  )
  self.prefill.push(prefill[0])
  self.prefill.push(scatter)
  for i in 1..<prefill.length() {
    self.prefill.push(prefill[i])
  }
  for i in 0..<decode.length() {
    self.decode.push(decode[i])
  }
  self.features = Some(features)
  self.feature_capacity = capacity.min(maximum_features.to_int64()).to_int()
  self.ready = true
  self.state = Prepared
}

///|
/// Call after vision completion and after the prior text queue retires. Two metadata
/// uploads per request; features stay on device. Retry overwrites the whole map.
pub fn Frame::bind_prompt(
  self : Self,
  context : @device.Context,
  prompt : ImagePrompt,
  features : @vision.FeatureView,
) -> Unit raise {
  guard self.ready &&
    self.state is (Prepared | Failed) &&
    features.allocation().same_storage(self.features.unwrap()) &&
    features.rows() == prompt.features &&
    features.rows() <= self.feature_capacity &&
    prompt.feature_map.length().to_int64() * 4L <=
    self.host_map.length().to_int64() else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  self.host_map.fill(b'\xff')
  for i in 0..<prompt.feature_map.length() {
    let word = prompt.feature_map[i].to_uint()
    for lane in 0..<4 {
      self.host_map[i * 4 + lane] = ((word >> (lane * 8)) & 255U).to_byte()
    }
  }
  let count = features.rows().to_uint()
  for lane in 0..<4 {
    self.host_count[lane] = ((count >> (lane * 8)) & 255U).to_byte()
  }
  self.state = Failed
  context.copy_from_fixed_host(
    self.allocations[0],
    self.host_map,
    0L,
    0L,
    self.host_map.length().to_int64(),
  )
  context.copy_from_fixed_host(self.allocations[1], self.host_count, 0L, 0L, 4L)
  self.state = Prepared
}

///|
/// Disable insertion for a text-only request after the prior queue retires.
pub fn Frame::clear_prompt(
  self : Self,
  context : @device.Context,
) -> Unit raise {
  guard self.ready && self.state is (Prepared | Failed) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  self.host_count.fill(b'\x00')
  self.state = Failed
  context.copy_from_fixed_host(self.allocations[1], self.host_count, 0L, 0L, 4L)
  self.state = Prepared
}

///|
pub fn Frame::launches(
  self : Self,
  phase : @text.Phase,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  match phase {
    Prefill => self.prefill[:]
    Decode => self.decode[:]
  }
}

///|
pub fn Frame::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.ready = false
  self.prefill.clear()
  self.decode.clear()
  while self.functions.last() is Some(function) {
    function.close()
    ignore(self.functions.pop())
  }
  while self.allocations.last() is Some(allocation) {
    allocation.close()
    ignore(self.allocations.pop())
  }
  self.features = None
  self.state = Closed
}
