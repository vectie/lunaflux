///|
priv enum State {
  Fresh
  Failed
  Prepared
  Closed
}

///|
/// Prepared dependent sampling owns only row embeddings and function handles.
/// Hidden/logits, weights, RNG, confidence and token-chain ports are borrowed.
/// The parent queue executes every row in order and owns cancellation/completion.
pub struct SequentialMarkovFrame {
  priv source : @source.SequentialMarkovCudaSource
  priv functions : Array[@device.Function]
  priv buffers : Array[@device.Allocation]
  priv table : Array[@device.OrderedKernelLaunch]
  priv mut state : State
}

///|
pub fn SequentialMarkovFrame::new(
  source : @source.SequentialMarkovCudaSource,
) -> Self {
  { source, functions: [], buffers: [], table: [], state: Fresh, }
}

///|
pub fn SequentialMarkovFrame::source_bytes(self : Self) -> Bytes {
  self.source.source_bytes()
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
pub fn SequentialMarkovFrame::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  output_count~ : @device.Allocation,
  tokens~ : @device.Allocation,
  pre_normalized_hidden~ : @device.Allocation,
  logits~ : @device.Allocation,
  embedding_weight~ : @device.Allocation,
  bias_weight~ : @device.Allocation,
  confidence_weight~ : @device.Allocation,
  rng~ : @device.Allocation,
  confidence~ : @device.Allocation,
  maximum_workspace_bytes~ : Int64,
) -> Unit raise {
  guard self.state is Fresh else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let plan = self.source.plan()
  guard plan.workspace_bytes() <= maximum_workspace_bytes else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  let (_, _, _, rows) = plan.geometry()
  let (embedding_bytes, bias_bytes, confidence_bytes) = plan.weight_bytes()
  let spans = [
    (output_count, 4L, 4L),
    (tokens, (rows.to_int64() + 1L) * 4L, 4L),
    (pre_normalized_hidden, plan.hidden_bytes(), 2L),
    (logits, plan.logits_bytes(), 4L),
    (embedding_weight, embedding_bytes, 2L),
    (bias_weight, bias_bytes, 2L),
    (confidence_weight, confidence_bytes, 2L),
    (rng, 16L, 8L),
    (confidence, rows.to_int64() * 4L, 4L),
  ]
  for (b, bytes, alignment) in spans {
    context.validate_allocation_region(
      b,
      offset=0L,
      byte_count=bytes,
      alignment~,
    )
  }
  // These writable planes cannot share storage with a read-only input. In
  // particular tokens must retain the previous row while sampling the next.
  for writable in [tokens, logits, confidence] {
    for
      read_port in [
        output_count, pre_normalized_hidden, embedding_weight, bias_weight, confidence_weight,
        rng,
      ] {
      guard !writable.same_storage(read_port) else {
        raise @precision.PrecisionPlanError::IllegalPlacement
      }
    }
  }
  guard !tokens.same_storage(logits) &&
    !tokens.same_storage(confidence) &&
    !logits.same_storage(confidence) else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let p = spans.map(span => {
    let (b, bytes, alignment) = span
    region(b, bytes, alignment)
  })
  self.state = Failed
  self.buffers.push(context.allocate(plan.embedding_frame_bytes()))
  self.buffers.push(context.allocate(rows.to_int64() * 4L))
  let indices = FixedArray::make(rows * 4, b'\x00')
  for row in 0..<rows {
    for byte in 0..<4 {
      indices[row * 4 + byte] = (row.to_uint() >> (byte * 8)).to_byte()
    }
  }
  self.buffers[1].copy_from_host(
    Bytes::from_array(indices),
    0L,
    0L,
    rows.to_int64() * 4L,
  )
  let embeddings = region(self.buffers[0], plan.embedding_frame_bytes(), 2L)
  for stage in 0..<4 {
    self.functions.push(
      aot.load_function(@utf8.encode(self.source.symbol(stage))),
    )
  }
  for row in 0..<rows {
    let row_arg = @device.KernelArgument::device_region(
      self.buffers[1],
      offset=row.to_int64() * 4L,
      byte_count=4L,
      alignment=4L,
    )
    let args = [
      [p[0], p[1], p[4], embeddings, row_arg],
      [p[0], p[1], embeddings, p[5], p[3], row_arg],
      [p[0], p[3], p[7], p[1], row_arg],
      [p[0], p[1], p[2], embeddings, p[6], p[8], row_arg],
    ]
    for stage in 0..<4 {
      let (blocks, threads) = self.source.launch(stage)
      self.table.push(
        @device.OrderedKernelLaunch::new(
          self.functions[stage],
          @device.KernelLaunchDimensions::new(
            grid_x=blocks,
            grid_y=1,
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
  self.state = Prepared
}

///|
pub fn SequentialMarkovFrame::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.table[:]
}

///|
pub fn SequentialMarkovFrame::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.table.clear()
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
