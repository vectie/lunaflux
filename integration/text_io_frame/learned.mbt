///|
/// Prepared learned residual-stream reduction and ordinary text egress. The
/// caller borrows its four suffix launches into the decoder's queue. No extra
/// submission/completion, host readback or token-step allocation is introduced.
pub struct LearnedTextIoFrame {
  priv plan : @precision.LearnedTextIoPrecision
  priv text : TextIoFrame
  priv source : Bytes
  priv prefix : String
  priv buffers : Array[@device.Allocation]
  priv functions : Array[@device.Function]
  priv after : Array[@device.OrderedKernelLaunch]
  priv mut state : State
}

///|
pub fn LearnedTextIoFrame::new(
  plan : @precision.LearnedTextIoPrecision,
  symbol_prefix~ : String,
) -> Self raise {
  {
    plan,
    text: TextIoFrame::new(
      @source.text_io_source(plan.text(), symbol_prefix~),
      boundary=Egress,
    ),
    source: @source.learned_text_io_source(plan, symbol_prefix~),
    prefix: symbol_prefix,
    buffers: [],
    functions: [],
    after: [],
    state: Fresh,
  }
}

///|
pub fn LearnedTextIoFrame::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn LearnedTextIoFrame::workspace_bytes(self : Self) -> Int64 {
  self.plan.workspace_bytes()
}

///|
/// All borrowed weights/ports must outlive the borrowing queue. Preparation
/// validates spans before allocating, and partial preparation is closeable.
pub fn LearnedTextIoFrame::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  counts~ : @device.Allocation,
  output_count~ : @device.Allocation,
  selected_rows~ : @device.Allocation,
  output_residual~ : @device.Allocation,
  function_weight~ : @device.Allocation,
  base~ : @device.Allocation,
  scale~ : @device.Allocation,
  norm_weight~ : @device.Allocation,
  head_weight~ : @device.Allocation,
  sampled_tokens~ : @device.Allocation,
  maximum_workspace_bytes~ : Int64,
) -> Unit raise {
  guard self.state is Fresh && self.workspace_bytes() <= maximum_workspace_bytes else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.state = Failed
  let (function_bytes, base_bytes, scale_bytes) = self.plan.control_weight_bytes()
  let counts_arg = region(counts, 20L, 4L)
  let input = region(output_residual, self.plan.input_bytes(), 2L)
  let function_arg = region(function_weight, function_bytes, 4L)
  let base_arg = region(base, base_bytes, 4L)
  let scale_arg = region(scale, scale_bytes, 4L)
  let (_, hidden, _, _, outputs) = self.plan.text().geometry()
  ignore(region(output_count, 4L, 4L))
  ignore(region(selected_rows, outputs.to_int64() * 4L, 4L))
  ignore(region(norm_weight, hidden.to_int64() * 2L, 2L))
  ignore(region(head_weight, self.plan.text().table_bytes(), 2L))
  ignore(region(sampled_tokens, outputs.to_int64() * 4L, 4L))
  let collapsed = context.allocate(self.plan.collapse_bytes())
  self.buffers.push(collapsed)
  let function = aot.load_function(@utf8.encode(self.prefix + "_stream_reduce"))
  self.functions.push(function)
  let (_, _, _, rows, _) = self.plan.text().geometry()
  self.after.push(
    @device.OrderedKernelLaunch::new(
      function,
      @device.KernelLaunchDimensions::new(
        grid_x=rows,
        grid_y=1,
        grid_z=1,
        block_x=256,
        block_y=1,
        block_z=1,
        shared_memory_bytes=0,
      ),
      @device.KernelArguments::new([
        counts_arg,
        input,
        function_arg,
        base_arg,
        scale_arg,
        region(collapsed, self.plan.collapse_bytes(), 2L),
      ]),
    ),
  )
  self.text.prepare_egress(
    context,
    aot,
    counts~,
    output_count~,
    selected_rows~,
    output_residual=collapsed,
    norm_weight~,
    head_weight~,
    sampled_tokens~,
    maximum_workspace_bytes=self.plan.text().workspace_bytes(),
  )
  self.after.append(self.text.suffix())
  self.state = Prepared
}

///|
pub fn LearnedTextIoFrame::suffix(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.after[:]
}

///|
pub fn LearnedTextIoFrame::logits(self : Self) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.text.logits()
}

///|
/// Close the borrowing queue before this owner, and this owner before weights.
pub fn LearnedTextIoFrame::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.after.clear()
  self.text.close()
  while self.functions.last() is Some(function) {
    function.close()
    ignore(self.functions.pop())
  }
  while self.buffers.last() is Some(buffer) {
    buffer.close()
    ignore(self.buffers.pop())
  }
  self.state = Closed
}
