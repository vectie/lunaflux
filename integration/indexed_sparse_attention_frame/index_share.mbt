///|
/// A consumer performs only sparse attention using an earlier layer's
/// device-resident selection. It owns no pool/index scratch and does not rerun
/// the selector. The producer's selection must outlive every consumer queue.
pub fn IndexedSparseAttentionFrame::for_shared_indices(
  source : @source.IndexedSparseAttentionCudaSource,
) -> Self {
  {
    source,
    buffers: [],
    functions: [],
    launches: [],
    shared: true,
    selection: None,
    status: Fresh,
  }
}

///|
fn IndexedSparseAttentionFrame::attention_arguments(
  self : Self,
  query_counts : @device.Allocation,
  history_counts : @device.Allocation,
  query_positions : @device.Allocation,
  history_valid : @device.Allocation,
  queries : @device.Allocation,
  keys : @device.Allocation,
  values : @device.Allocation,
  selected_indices : @device.Allocation,
  output : @device.Allocation,
) -> Array[@device.KernelArgument] raise @device.DeviceError {
  let plan = self.source.plan()
  let (rows, history) = plan.capacity()
  let (qh, kvh, qkd, vd) = plan.attention_geometry()
  let r = rows.to_int64()
  let h = history.to_int64()
  [
    region(query_counts, 20L, 4L),
    region(query_positions, r * 4L, 4L),
    @device.KernelArgument::device_region(
      history_counts,
      offset=12L,
      byte_count=4L,
      alignment=4L,
    ),
    region(history_valid, h * 4L, 4L),
    region(queries, r * qh.to_int64() * qkd.to_int64() * 2L, 2L),
    region(keys, h * kvh.to_int64() * qkd.to_int64() * 2L, 2L),
    region(values, h * kvh.to_int64() * vd.to_int64() * 2L, 2L),
    region(selected_indices, plan.frame_bytes()[4], 4L),
    region(output, r * qh.to_int64() * vd.to_int64() * 2L, 2L),
  ]
}

///|
/// Bind one IndexShare consumer without owning or copying selected indices.
/// The immutable plan fixes their row width; the model's layer plan chooses
/// which producer is reused. All histories and output remain caller-owned.
pub fn IndexedSparseAttentionFrame::prepare_shared_indices(
  self : Self,
  aot : @device.Module,
  query_counts~ : @device.Allocation,
  history_counts~ : @device.Allocation,
  query_positions~ : @device.Allocation,
  history_valid~ : @device.Allocation,
  queries~ : @device.Allocation,
  keys~ : @device.Allocation,
  values~ : @device.Allocation,
  selected_indices~ : @device.Allocation,
  output~ : @device.Allocation,
) -> Unit raise {
  guard self.status is Fresh && self.shared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.status = Failed
  let arguments = self.attention_arguments(
    query_counts, history_counts, query_positions, history_valid, queries, keys,
    values, selected_indices, output,
  )
  let function = aot.load_function(@utf8.encode(self.source.symbol(2)))
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
      @device.KernelArguments::new(arguments),
    ),
  )
  self.selection = Some(selected_indices)
  self.status = Prepared
}
