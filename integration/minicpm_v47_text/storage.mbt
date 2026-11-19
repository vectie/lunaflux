///|
priv struct StorageObjects {
  metadata : Metadata
  states : @decoder.StateArenas
  buffers : @decoder.Buffers
  sampled_tokens : @device.Allocation
}

///|
priv enum StorageState {
  Fresh
  Failed
  Prepared
  Closed
}

///|
/// Owns the complete mutable allocation set for one prepared text profile.
/// Close executors/Frame before Storage, then release the borrowed context.
/// Device allocation does not initialize request metadata or clear state.
pub struct Storage {
  priv program : Program
  priv allocations : Array[@device.Allocation]
  priv mut objects : StorageObjects?
  priv mut state : StorageState
}

///|
pub fn Storage::new(program : Program) -> Self {
  { program, allocations: [], objects: None, state: Fresh, }
}

///|
/// Admit the entire mutable allocation budget before allocating. This ceiling
/// excludes weights/modules and the frame's separate text output workspace.
/// Close this owner after partial failure before releasing the context.
pub fn Storage::prepare(
  self : Self,
  context : @device.Context,
  maximum_device_bytes~ : Int64,
) -> Unit raise @device.DeviceError {
  guard self.state is Fresh &&
    self.program.storage_budget().owned_bytes <= maximum_device_bytes else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  self.state = Failed
  let allocate : (Int64) -> @device.Allocation raise @device.DeviceError = fn(
    bytes,
  ) raise @device.DeviceError {
    let allocation = context.allocate(bytes)
    self.allocations.push(allocation)
    allocation
  }
  let budget = self.program.decoder.budget()
  let buffers : @decoder.Buffers = {
    hidden: allocate(budget.activation_bytes),
    normalized: allocate(budget.activation_bytes),
    branch_output: allocate(budget.activation_bytes),
    branch_workspace: allocate(budget.branch_workspace_bytes),
    moe_workspace: allocate(budget.moe_workspace_bytes),
  }
  let states : @decoder.StateArenas = {
    convolution: allocate(budget.convolution_state_bytes),
    recurrent: allocate(budget.recurrent_state_bytes),
    key_cache: allocate(budget.key_cache_bytes),
    value_cache: allocate(budget.value_cache_bytes),
  }
  let attention = self.program.decoder.attention_program()
  let recurrent = self.program.decoder.recurrent_program()
  let (_, _, _, rows, outputs) = self.program.text.plan().geometry()
  let attention_metadata : @attention.Metadata = {
    counts: allocate(attention.region(Counts).byte_count),
    canvas_positions: allocate(attention.region(CanvasPositions).byte_count),
    causal_positions: allocate(attention.region(CausalPositions).byte_count),
    row_offsets: allocate(attention.region(RowOffsets).byte_count),
    sequence_lengths: allocate(attention.region(SequenceLengths).byte_count),
    page_offsets: allocate(attention.region(PageOffsets).byte_count),
    physical_pages: allocate(attention.region(PhysicalPages).byte_count),
  }
  let metadata : Metadata = {
    decoder: {
      attention: attention_metadata,
      request_slots: allocate(recurrent.region(RequestSlots).byte_count),
      reset: allocate(recurrent.region(Reset).byte_count),
    },
    token_ids: allocate(rows.to_int64() * 4L),
    output_count: allocate(4L),
    selected_rows: allocate(outputs.to_int64() * 4L),
  }
  self.objects = Some({
    metadata,
    states,
    buffers,
    sampled_tokens: allocate(outputs.to_int64() * 4L),
  })
  self.state = Prepared
}

///|
fn Storage::borrow(self : Self) -> StorageObjects raise @device.DeviceError {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  self.objects.unwrap()
}

///|
pub fn Storage::metadata(self : Self) -> Metadata raise @device.DeviceError {
  self.borrow().metadata
}

///|
pub fn Storage::states(
  self : Self,
) -> @decoder.StateArenas raise @device.DeviceError {
  self.borrow().states
}

///|
pub fn Storage::buffers(
  self : Self,
) -> @decoder.Buffers raise @device.DeviceError {
  self.borrow().buffers
}

///|
pub fn Storage::sampled_tokens(
  self : Self,
) -> @device.Allocation raise @device.DeviceError {
  self.borrow().sampled_tokens
}

///|
pub fn Storage::close(self : Self) -> Unit raise @device.DeviceError {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.objects = None
  while self.allocations.last() is Some(allocation) {
    allocation.close()
    ignore(self.allocations.pop())
  }
  self.state = Closed
}
