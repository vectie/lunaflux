///|
priv enum State {
  Fresh
  Prepared
  Failed
  Closed
}

///|
/// Owns projected pooling, index selection and sparse-attention intermediates.
/// Histories and query frames are borrowed independently. The caller's queue
/// must close before this frame; no second queue or completion is created.
pub struct IndexedSparseAttentionFrame {
  priv source : @source.IndexedSparseAttentionCudaSource
  priv buffers : Array[@device.Allocation]
  priv functions : Array[@device.Function]
  priv launches : Array[@device.OrderedKernelLaunch]
  priv shared : Bool
  priv mut selection : @device.Allocation?
  priv mut status : State
}

///|
pub fn IndexedSparseAttentionFrame::new(
  source : @source.IndexedSparseAttentionCudaSource,
) -> Self {
  {
    source,
    buffers: [],
    functions: [],
    launches: [],
    shared: false,
    selection: None,
    status: Fresh,
  }
}

///|
pub fn IndexedSparseAttentionFrame::source_bytes(self : Self) -> Bytes {
  self.source.source_bytes()
}

///|
pub fn IndexedSparseAttentionFrame::workspace_bytes(self : Self) -> Int64 {
  if self.shared {
    0L
  } else {
    self.source.plan().workspace_bytes()
  }
}

///|
pub fn IndexedSparseAttentionFrame::attention_plan(
  self : Self,
) -> @precision.IndexedSparseAttentionPrecision {
  self.source.plan()
}

///|
fn region(
  allocation : @device.Allocation,
  bytes : Int64,
  alignment : Int64,
) -> @device.KernelArgument raise @device.DeviceError {
  @device.KernelArgument::device_region(
    allocation,
    offset=0L,
    byte_count=bytes,
    alignment~,
  )
}

///|
/// Counts use the established five-I32 frame ABI, query count at offset 12.
/// History counts hold the retained key count there instead. Both describe one
/// request; scheduler-owned histories must retain absolute positions and valid
/// masks. This frame does not substitute batch-local rows for cache positions.
pub fn IndexedSparseAttentionFrame::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  maximum_workspace_bytes~ : Int64,
  query_counts~ : @device.Allocation,
  history_counts~ : @device.Allocation,
  query_positions~ : @device.Allocation,
  index_queries~ : @device.Allocation,
  index_head_weights~ : @device.Allocation,
  index_keys~ : @device.Allocation,
  pool_gate_scores~ : @device.Allocation,
  pool_ape~ : @device.Allocation,
  history_valid~ : @device.Allocation,
  queries~ : @device.Allocation,
  keys~ : @device.Allocation,
  values~ : @device.Allocation,
  output~ : @device.Allocation,
) -> Unit raise {
  guard self.status is Fresh &&
    !self.shared &&
    self.workspace_bytes() <= maximum_workspace_bytes else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.status = Failed
  let plan = self.source.plan()
  let (rows, history) = plan.capacity()
  let (ih, id, pool, _, _) = plan.index_geometry()
  let r = rows.to_int64()
  let h = history.to_int64()
  let sizes = plan.frame_bytes()
  for bytes in sizes {
    self.buffers.push(context.allocate(bytes))
  }
  let buffers = self.buffers
  let selected = region(buffers[4], sizes[4], 4L)
  let arguments = [
    [
      region(history_counts, 20L, 4L),
      region(index_keys, h * id.to_int64() * 2L, 2L),
      region(pool_gate_scores, h * id.to_int64() * 2L, 2L),
      region(history_valid, h * 4L, 4L),
      region(pool_ape, pool.to_int64() * id.to_int64() * 2L, 2L),
      region(buffers[0], sizes[0], 2L),
      region(buffers[1], sizes[1], 4L),
      region(buffers[2], sizes[2], 4L),
      region(buffers[3], sizes[3], 4L),
    ],
    [
      region(query_counts, 20L, 4L),
      region(query_positions, r * 4L, 4L),
      region(index_queries, r * ih.to_int64() * id.to_int64() * 2L, 2L),
      region(index_head_weights, r * ih.to_int64() * 2L, 2L),
      region(buffers[0], sizes[0], 2L),
      region(buffers[1], sizes[1], 4L),
      region(buffers[2], sizes[2], 4L),
      region(buffers[3], sizes[3], 4L),
      region(history_valid, h * 4L, 4L),
      selected,
    ],
    self.attention_arguments(
      query_counts,
      history_counts,
      query_positions,
      history_valid,
      queries,
      keys,
      values,
      buffers[4],
      output,
    ),
  ]
  for stage in 0..<3 {
    let function = aot.load_function(@utf8.encode(self.source.symbol(stage)))
    self.functions.push(function)
    self.launches.push(
      @device.OrderedKernelLaunch::new(
        function,
        @device.KernelLaunchDimensions::new(
          grid_x=1,
          grid_y=1,
          grid_z=1,
          block_x=1,
          block_y=1,
          block_z=1,
          shared_memory_bytes=0,
        ),
        @device.KernelArguments::new(arguments[stage]),
      ),
    )
  }
  self.status = Prepared
  self.selection = Some(buffers[4])
}

///|
pub fn IndexedSparseAttentionFrame::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard self.status is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}

///|
/// Share this device allocation with an IndexShare consumer, never a host copy.
/// Its borrowing queue and consumers must be closed before the producer frame.
pub fn IndexedSparseAttentionFrame::selected_indices(
  self : Self,
) -> @device.Allocation raise @device.DeviceError {
  guard self.status is Prepared && self.selection is Some(selection) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  selection
}

///|
pub fn IndexedSparseAttentionFrame::close(
  self : Self,
) -> Unit raise @device.DeviceError {
  if self.status is Closed {
    return
  }
  self.status = Failed
  self.launches.clear()
  self.selection = None
  while self.functions.last() is Some(function) {
    function.close()
    ignore(self.functions.pop())
  }
  while self.buffers.last() is Some(buffer) {
    buffer.close()
    ignore(self.buffers.pop())
  }
  self.status = Closed
}
