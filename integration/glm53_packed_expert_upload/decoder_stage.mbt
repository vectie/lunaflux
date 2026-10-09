///|
/// Prepared model stage: owns text weights only at model boundaries, connected
/// decoder layers, and distinct input/output residuals. Request ports/module/
/// stream are borrowed. A complete range can bind worker delivery; a partial
/// stage exposes residuals for a composed pipeline, never a fake model token.
pub struct GlmDecoderStage {
  priv range : @pipeline.DecoderStageRange
  priv decoder : GlmDecoderSlice
  priv text : GlmTextIo
  priv residual_bytes : Int64
  priv maximum_rows : Int
  priv maximum_history : Int
  priv vocabulary : Int
  priv residuals : Array[@device.Allocation]
  priv mut counts : @device.Allocation?
  priv mut activation_leased : Bool
  priv mut active_rows : Int
  priv mut submission_epoch : UInt64
  priv mut status : RecurrentLayerState
}

///|
pub fn GlmDecoderStage::new(
  spec : @glm53_spec.Glm53FlashSpec,
  range : @pipeline.DecoderStageRange,
  maximum_rows~ : Int,
  maximum_history~ : Int,
  index_layer_norm_epsilon~ : Double,
  maximum_layer_state_bytes~ : Int64,
  maximum_layer_workspace_bytes~ : Int64,
  maximum_layer_expert_bytes~ : Int64,
  maximum_text_workspace_bytes~ : Int64,
  accuracy~ : @precision_ir.Accuracy,
) -> Self raise {
  guard range.layer_count() == spec.layer_count() &&
    maximum_history >= maximum_rows else {
    raise @precision_ir.PrecisionPlanError::IllegalPlacement
  }
  let (first_layer, end_layer) = range.layer_range()
  let text_plan = flash_text_io_plan(
    spec,
    maximum_rows~,
    maximum_output_rows=1,
    maximum_workspace_bytes=maximum_text_workspace_bytes,
    boundary=range.text_boundary(),
  )
  {
    range,
    decoder: GlmDecoderSlice::new(
      spec,
      first_layer~,
      end_layer~,
      maximum_rows~,
      maximum_sequences=1,
      cache_slots=1,
      maximum_history~,
      index_layer_norm_epsilon~,
      maximum_layer_state_bytes~,
      maximum_layer_workspace_bytes~,
      maximum_layer_expert_bytes~,
      accuracy~,
    ),
    text: GlmTextIo::new(
      spec,
      maximum_rows~,
      maximum_output_rows=1,
      maximum_workspace_bytes=maximum_text_workspace_bytes,
      boundary=range.text_boundary(),
    ),
    residual_bytes: text_plan.residual_bytes(),
    maximum_rows,
    maximum_history,
    vocabulary: spec.vocabulary_size(),
    counts: None,
    activation_leased: false,
    active_rows: 0,
    submission_epoch: 0UL,
    residuals: [],
    status: Fresh,
  }
}

///|
pub fn GlmDecoderStage::layer_range(self : Self) -> (Int, Int) {
  self.range.layer_range()
}

///|
pub fn GlmDecoderStage::text_boundary(
  self : Self,
) -> @precision_ir.TextBoundary {
  self.range.text_boundary()
}

///|
pub fn GlmDecoderStage::source_bytes(self : Self) -> Bytes {
  self.text.source_bytes() + b"\n" + self.decoder.source_bytes()
}

///|
/// Includes all owned weights, state, workspace and two boundary residuals.
/// Caller separately reserves borrowed module/stream/request ports and host IO.
pub fn GlmDecoderStage::required_device_bytes(
  self : Self,
  checkpoint : @precision_materialize.PackedCheckpoint,
) -> Int64 raise {
  self.decoder.required_device_bytes(checkpoint) +
  self.text.required_device_bytes(checkpoint) +
  2L * self.residual_bytes
}

///|
/// Budget the entire stage before its first allocation/upload. The text prefix
/// and suffix join the same decoder queue. Boundary residuals are never aliased.
pub fn GlmDecoderStage::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  stream : @device.Stream,
  checkpoint : @precision_materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  ports : @serial_worker.SerialDecoderWorker,
  maximum_device_bytes~ : Int64,
  chunk_bytes~ : Int,
  limits~ : @streaming_safetensors.StreamingSafetensorsLimits,
) -> Unit raise {
  guard self.status is Fresh else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let (rows, history, vocabulary) = ports.geometry()
  guard rows == self.maximum_rows &&
    history <= self.maximum_history &&
    vocabulary == self.vocabulary else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let decoder_bytes = self.decoder.required_device_bytes(checkpoint)
  let text_bytes = self.text.required_device_bytes(checkpoint)
  guard maximum_device_bytes >= 2L * self.residual_bytes &&
    text_bytes <= maximum_device_bytes - 2L * self.residual_bytes &&
    decoder_bytes <=
    maximum_device_bytes - 2L * self.residual_bytes - text_bytes else {
    raise @precision_ir.PrecisionPlanError::MemoryBudgetExceeded
  }
  // Resolve all borrowed inputs before acquiring stage storage.
  let counts = ports.input(Counts)
  let ids = ports.input(TokenIds)
  let offsets = ports.input(SequenceOffsets)
  let slots = ports.input(SequenceSlots)
  let resets = ports.input(RecurrentResets)
  let sparse_reset = ports.input(SparseReset)
  let output_count = ports.input(OutputCount)
  let selected = ports.input(SelectedRows)
  let sampled = ports.sampled_tokens()
  self.status = Failed
  self.residuals.push(context.allocate(self.residual_bytes))
  self.residuals.push(context.allocate(self.residual_bytes))
  self.text.prepare(
    context,
    aot,
    checkpoint,
    root,
    maximum_device_bytes=text_bytes,
    chunk_bytes~,
    limits~,
    counts~,
    token_ids=ids,
    output_count~,
    selected_rows=selected,
    input_residual=self.residuals[0],
    output_residual=self.residuals[1],
    sampled_tokens=sampled,
  )
  self.decoder.prepare(
    context,
    aot,
    stream,
    checkpoint,
    root,
    maximum_device_bytes=decoder_bytes,
    chunk_bytes~,
    limits~,
    counts~,
    sequence_offsets=offsets,
    sequence_slots=slots,
    reset_flags=resets,
    sparse_reset_flag=sparse_reset,
    hidden=self.residuals[0],
    output=self.residuals[1],
    prefix_launches=self.text.prefix(),
    suffix_launches=self.text.suffix(),
  )
  self.counts = Some(counts)
  self.status = Ready
}

///|
/// Borrow residual ports only for startup-bound pipeline transfer owners. The
/// producer must retire before the consumer reads and before storage release.
pub fn GlmDecoderStage::input_residual(self : Self) -> @device.Allocation raise {
  guard self.status is Ready else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.residuals[0]
}

///|
pub fn GlmDecoderStage::output_residual(
  self : Self,
) -> @device.Allocation raise {
  guard self.status is Ready else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.residuals[1]
}

///|
pub fn GlmDecoderStage::launch_count(self : Self) -> Int {
  self.decoder.launch_count()
}

///|
pub fn GlmDecoderStage::append_descriptors(
  self : Self,
) -> ArrayView[@device.Allocation] {
  self.decoder.append_descriptors()
}

///|
pub fn GlmDecoderStage::submit(self : Self, live_rows~ : Int) -> Unit raise {
  guard self.status is Ready && !self.activation_leased else {
    raise @device.DeviceError::Busy(OrderedExecutorEnqueue)
  }
  guard live_rows > 0 &&
    live_rows <= self.maximum_rows &&
    self.submission_epoch < 0xffffffffffffffffUL else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  self.status = Failed
  self.decoder.submit(live_rows~)
  self.active_rows = live_rows
  self.submission_epoch += 1UL
  self.status = Active
}

///|
pub fn GlmDecoderStage::poll(self : Self) -> Bool raise {
  guard self.status is Active else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  self.status = Failed
  if self.decoder.poll() {
    self.status = Ready
    true
  } else {
    self.status = Active
    false
  }
}

///|
/// Rank error/sample retirement and activation DMA may both observe this one
/// submission. Once its queue has retired, neither observer polls it again.
fn GlmDecoderStage::poll_submission(self : Self) -> Bool raise {
  if self.status is Ready && self.active_rows > 0 {
    true
  } else {
    self.poll()
  }
}

///|
/// Close worker/transport leases first, then queue, text weights and residuals.
pub fn GlmDecoderStage::close(self : Self) -> Unit raise {
  if self.status is Closed {
    return
  }
  guard !self.activation_leased else {
    raise @device.DeviceError::Busy(OrderedExecutorClose)
  }
  self.status = Failed
  self.decoder.close()
  self.counts = None
  self.text.close()
  while self.residuals.last() is Some(buffer) {
    buffer.close()
    ignore(self.residuals.pop())
  }
  self.status = Closed
}
