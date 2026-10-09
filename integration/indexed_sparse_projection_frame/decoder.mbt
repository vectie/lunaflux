///|
/// Complete sparse attention sublayer. The caller owns the single executor;
/// all weights, input/output, reset/counts and pool APE remain borrowed.
pub struct IndexedSparseDecoderFrame {
  priv source : @source.IndexedSparseDecoderCudaSource
  priv projection : IndexedSparseProjectionFrame
  priv history : @sparse.IndexedSparseHistory
  priv attention : @sparse.IndexedSparseAttentionFrame
  priv envelope : @hyper.HyperConnectionFrame
  priv buffers : Array[@device.Allocation]
  priv functions : Array[@device.Function]
  priv launches : Array[@device.OrderedKernelLaunch]
  priv mut state : State
}

///|
pub fn IndexedSparseDecoderFrame::new(
  source : @source.IndexedSparseDecoderCudaSource,
) -> Self {
  {
    source,
    projection: IndexedSparseProjectionFrame::new(source.projection()),
    history: @sparse.IndexedSparseHistory::new(source.history()),
    attention: @sparse.IndexedSparseAttentionFrame::new(source.attention()),
    envelope: @hyper.HyperConnectionFrame::new(source.envelope()),
    buffers: [],
    functions: [],
    launches: [],
    state: Fresh,
  }
}

///|
pub fn IndexedSparseDecoderFrame::source_bytes(self : Self) -> Bytes {
  self.source.source_bytes()
}

///|
pub fn IndexedSparseDecoderFrame::workspace_bytes(self : Self) -> Int64 {
  self.source.plan().workspace_bytes()
}

///|
pub fn IndexedSparseDecoderFrame::state_bytes(self : Self) -> Int64 {
  self.history.state_bytes()
}

///|
pub fn IndexedSparseDecoderFrame::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  maximum_workspace_bytes~ : Int64,
  maximum_state_bytes~ : Int64,
  counts~ : @device.Allocation,
  reset_flag~ : @device.Allocation,
  hidden~ : @device.Allocation,
  output~ : @device.Allocation,
  projection_weights~ : ArrayView[@device.Allocation],
  pool_ape~ : @device.Allocation,
  output_weight~ : @device.Allocation,
  hyper_weights~ : ArrayView[@device.Allocation],
) -> Unit raise {
  guard self.state is Fresh &&
    projection_weights.length() == 12 &&
    hyper_weights.length() == 4 &&
    self.workspace_bytes() <= maximum_workspace_bytes &&
    self.state_bytes() <= maximum_state_bytes else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.state = Failed
  let plan = self.source.plan()
  let attention_output = context.allocate(plan.attention_output_bytes())
  self.buffers.push(attention_output)
  self.envelope.prepare(
    context,
    aot,
    counts~,
    input=hidden,
    output~,
    function_weight=hyper_weights[0],
    base=hyper_weights[1],
    scale=hyper_weights[2],
    norm_weight=hyper_weights[3],
    maximum_workspace_bytes=self.envelope.workspace_bytes(),
  )
  self.projection.prepare(
    context,
    aot,
    maximum_workspace_bytes=self.projection.workspace_bytes(),
    query_counts=counts,
    hidden=self.envelope.branch_input(),
    weights=projection_weights,
  )
  let sparse = self.projection.prepare_retained_attention(
    context,
    aot,
    self.history,
    self.attention,
    maximum_state_bytes~,
    maximum_attention_workspace_bytes=self.attention.workspace_bytes(),
    reset_flag~,
    pool_ape~,
    output=attention_output,
  )
  self.launches.append(self.envelope.prefix())
  self.launches.append(sparse)
  let function = aot.load_function(@utf8.encode(self.source.output_symbol()))
  self.functions.push(function)
  let (hidden_width, _, _) = plan.projection().geometry()
  let (rows, _) = plan.projection().attention().capacity()
  let out_bytes = rows.to_int64() * hidden_width.to_int64() * 2L
  self.launches.push(
    @device.OrderedKernelLaunch::new(
      function,
      @device.KernelLaunchDimensions::new(
        grid_x=((out_bytes / 2L + 255L) / 256L).to_int(),
        grid_y=1,
        grid_z=1,
        block_x=256,
        block_y=1,
        block_z=1,
        shared_memory_bytes=0,
      ),
      @device.KernelArguments::new([
        @device.KernelArgument::device_region(
          counts,
          offset=0L,
          byte_count=20L,
          alignment=4L,
        ),
        region(attention_output, plan.attention_output_bytes()),
        region(output_weight, plan.output_weight_bytes()),
        region(self.envelope.branch_output(), out_bytes),
      ]),
    ),
  )
  self.launches.append(self.envelope.suffix())
  self.state = Prepared
}

///|
pub fn IndexedSparseDecoderFrame::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}

///|
/// Borrowed device-side overflow descriptor. Output delivery must honor errors
/// before publishing a frame; this does not force a diagnostic host readback.
pub fn IndexedSparseDecoderFrame::append_descriptor(
  self : Self,
) -> @device.Allocation raise @device.DeviceError {
  self.history.append_descriptor()
}

///|
/// Close the borrowing executor first, including on partial preparation failure.
pub fn IndexedSparseDecoderFrame::close(
  self : Self,
) -> Unit raise @device.DeviceError {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.launches.clear()
  while self.functions.last() is Some(function) {
    function.close()
    ignore(self.functions.pop())
  }
  self.attention.close()
  self.history.close()
  self.projection.close()
  self.envelope.close()
  while self.buffers.last() is Some(buffer) {
    buffer.close()
    ignore(self.buffers.pop())
  }
  self.state = Closed
}
