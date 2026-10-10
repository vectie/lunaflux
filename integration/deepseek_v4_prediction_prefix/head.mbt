///|
/// Checkpoint-backed five-row prediction head. Borrows the base vocabulary
/// matrix, final prediction residual, counts, explicit RNG and output ports.
/// Owns compact prediction weights, learned-head and sequential-chain scratch.
pub struct DeepSeekPredictionHead {
  priv spec : @spec.DeepSeekV4ModelSpec
  priv text : @text_frame.LearnedTextIoFrame
  priv markov : @markov_frame.SequentialMarkovFrame
  priv workspace : Int64
  priv weights : @upload.CheckpointDeviceWeights
  priv buffers : Array[@device.Allocation]
  priv table : Array[@device.OrderedKernelLaunch]
  priv mut state : AttentionState
}

///|
pub fn DeepSeekPredictionHead::new(
  spec : @spec.DeepSeekV4ModelSpec,
  temperature~ : Double,
  maximum_workspace_bytes~ : Int64,
) -> Self raise {
  let text_plan = prediction_text_head_plan(spec, maximum_workspace_bytes~)
  let markov_plan = prediction_markov_plan(
    spec,
    temperature~,
    maximum_workspace_bytes~,
  )
  let workspace = text_plan.workspace_bytes() +
    markov_plan.workspace_bytes() +
    4L +
    spec.dspark_block_size().to_int64() * 4L
  guard workspace <= maximum_workspace_bytes else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  {
    spec,
    text: @text_frame.LearnedTextIoFrame::new(
      text_plan,
      symbol_prefix="deepseek_prediction_head",
    ),
    markov: @markov_frame.SequentialMarkovFrame::new(
      @markov_source.sequential_markov_source(
        markov_plan,
        symbol_prefix="deepseek_prediction_markov",
      ),
    ),
    workspace,
    weights: @upload.CheckpointDeviceWeights::new(),
    buffers: [],
    table: [],
    state: Fresh,
  }
}

///|
pub fn DeepSeekPredictionHead::source_bytes(self : Self) -> Bytes {
  self.text.source_bytes() + b"\n" + self.markov.source_bytes()
}

///|
pub fn DeepSeekPredictionHead::required_device_bytes(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
) -> Int64 raise {
  bind_prediction_head_weights(checkpoint, self.spec).fold(
    init=self.workspace,
    (bytes, w) => bytes + w.buffer_bytes(),
  )
}

///|
/// tokens[0] must contain the seed; counts describes all five draft rows.
/// RNG is two U64 words (seed, sequence). Positive-temperature replay with the
/// same words is reproducible; callers advance sequence between predictions.
pub fn DeepSeekPredictionHead::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  counts~ : @device.Allocation,
  residual~ : @device.Allocation,
  vocabulary_weight~ : @device.Allocation,
  tokens~ : @device.Allocation,
  confidence~ : @device.Allocation,
  rng~ : @device.Allocation,
  maximum_device_bytes~ : Int64,
  chunk_bytes~ : Int,
  limits~ : @streaming.StreamingSafetensorsLimits,
) -> Unit raise {
  guard self.state is Fresh && chunk_bytes > 0 else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  guard self.required_device_bytes(checkpoint) <= maximum_device_bytes else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  let rows = self.spec.dspark_block_size()
  for
    (b, bytes, alignment) in [
      (counts, 20L, 4L),
      (
        residual,
        rows.to_int64() *
        self.spec.hidden_size().to_int64() *
        self.spec.hyper_connection_width().to_int64() *
        2L,
        2L,
      ),
      (
        vocabulary_weight,
        self.spec.vocabulary_size().to_int64() *
        self.spec.hidden_size().to_int64() *
        2L,
        2L,
      ),
      (tokens, (rows.to_int64() + 1L) * 4L, 4L),
      (confidence, rows.to_int64() * 4L, 4L),
      (rng, 16L, 8L),
    ] {
    context.validate_allocation_region(
      b,
      offset=0L,
      byte_count=bytes,
      alignment~,
    )
  }
  for writable in [tokens, confidence] {
    for input_port in [counts, residual, vocabulary_weight, rng] {
      guard !writable.same_storage(input_port) else {
        raise @precision.PrecisionPlanError::IllegalPlacement
      }
    }
  }
  guard !tokens.same_storage(confidence) else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  self.state = Failed
  self.weights.upload(
    context,
    checkpoint,
    root,
    bind_prediction_head_weights(checkpoint, self.spec),
    maximum_device_bytes=maximum_device_bytes - self.workspace,
    chunk_bytes~,
    limits~,
  )
  self.buffers.push(context.allocate(4L))
  self.buffers.push(context.allocate(rows.to_int64() * 4L))
  let indices = FixedArray::make(rows * 4, b'\x00')
  let count = FixedArray::make(4, b'\x00')
  for byte in 0..<4 {
    count[byte] = (rows.to_uint() >> (byte * 8)).to_byte()
  }
  for row in 0..<rows {
    for byte in 0..<4 {
      indices[row * 4 + byte] = (row.to_uint() >> (byte * 8)).to_byte()
    }
  }
  self.buffers[0].copy_from_host(Bytes::from_array(count), 0L, 0L, 4L)
  self.buffers[1].copy_from_host(
    Bytes::from_array(indices),
    0L,
    0L,
    rows.to_int64() * 4L,
  )
  self.text.prepare(
    context,
    aot,
    counts~,
    output_count=self.buffers[0],
    selected_rows=self.buffers[1],
    output_residual=residual,
    norm_weight=self.weights.allocation(0),
    function_weight=self.weights.allocation(1),
    base=self.weights.allocation(2),
    scale=self.weights.allocation(3),
    head_weight=vocabulary_weight,
    sampled_tokens=tokens,
    maximum_workspace_bytes=self.text.workspace_bytes(),
  )
  self.markov.prepare(
    context,
    aot,
    output_count=self.buffers[0],
    tokens~,
    pre_normalized_hidden=self.text.pre_normalized_hidden(),
    logits=self.text.logits(),
    embedding_weight=self.weights.allocation(4),
    bias_weight=self.weights.allocation(5),
    confidence_weight=self.weights.allocation(6),
    rng~,
    confidence~,
    maximum_workspace_bytes=self.workspace -
      self.text.workspace_bytes() -
      4L -
      rows.to_int64() * 4L,
  )
  self.table.append(self.text.projection_suffix())
  self.table.append(self.markov.launches())
  self.state = Prepared
}

///|
pub fn DeepSeekPredictionHead::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.table[:]
}

///|
pub fn DeepSeekPredictionHead::logits(self : Self) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.text.logits()
}

///|
pub fn DeepSeekPredictionHead::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.table.clear()
  self.markov.close()
  self.text.close()
  self.weights.close()
  while self.buffers.last() is Some(b) {
    b.close()
    ignore(self.buffers.pop())
  }
  self.state = Closed
}
