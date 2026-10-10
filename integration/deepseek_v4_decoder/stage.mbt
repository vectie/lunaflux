///|
/// A contiguous base-model stage with text effects only at the model edges.
/// Compact checkpoint weights/state and two residual boundaries are owned;
/// token tables are also stage-owned; request ports, index offsets, AOT module
/// and stream are borrowed.
pub struct DeepSeekDecoderStage {
  priv spec : @spec.DeepSeekV4ModelSpec
  priv range : @pipeline.DecoderStageRange
  priv decoder : DeepSeekDecoderSlice
  priv ingress : @text.DeepSeekTextIngress?
  priv egress : @text.DeepSeekTextIo?
  priv maximum_rows : Int
  priv maximum_context : Int
  priv vocabulary : Int
  priv residual_bytes : Int64
  priv token_table_count : Int
  priv token_tables : @upload.CheckpointDeviceIndexTables
  priv needs_index : Bool
  priv residuals : Array[@device.Allocation]
  priv mut counts : @device.Allocation?
  priv mut state : SliceState
}

///|
pub fn DeepSeekDecoderStage::new(
  spec : @spec.DeepSeekV4ModelSpec,
  range : @pipeline.DecoderStageRange,
  maximum_rows~ : Int,
  maximum_context~ : Int,
  maximum_layer_state_bytes~ : Int64,
  maximum_layer_workspace_bytes~ : Int64,
  maximum_layer_expert_bytes~ : Int64,
  maximum_text_workspace_bytes~ : Int64,
  accuracy~ : @precision.Accuracy,
) -> Self raise {
  guard range.layer_count() == spec.layer_count() else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let (first_layer, end_layer) = range.layer_range()
  let boundary = range.text_boundary()
  let decoder = DeepSeekDecoderSlice::new(
    spec,
    first_layer~,
    end_layer~,
    maximum_rows~,
    maximum_context~,
    maximum_layer_state_bytes~,
    maximum_layer_workspace_bytes~,
    maximum_layer_expert_bytes~,
    accuracy~,
  )
  let ingress = if boundary.has_ingress() {
    Some(@text.DeepSeekTextIngress::new(spec, maximum_rows~))
  } else {
    None
  }
  let egress = if boundary.has_egress() {
    Some(
      @text.DeepSeekTextIo::new(
        spec,
        maximum_rows~,
        maximum_output_rows=1,
        maximum_workspace_bytes=maximum_text_workspace_bytes,
      ),
    )
  } else {
    None
  }
  let needs_index = for layer in first_layer..<end_layer {
    if spec.compression_ratio(layer) == Some(4) {
      break true
    }
  } nobreak {
    false
  }
  {
    spec,
    range,
    decoder,
    ingress,
    egress,
    maximum_rows,
    maximum_context,
    vocabulary: spec.vocabulary_size(),
    residual_bytes: maximum_rows.to_int64() *
    spec.hidden_size().to_int64() *
    spec.hyper_connection_width().to_int64() *
    2L,
    token_table_count: (end_layer.min(spec.hash_layer_count()) - first_layer).max(
      0,
    ),
    token_tables: @upload.CheckpointDeviceIndexTables::new(),
    needs_index,
    residuals: [],
    counts: None,
    state: Fresh,
  }
}

///|
pub fn DeepSeekDecoderStage::layer_range(self : Self) -> (Int, Int) {
  self.range.layer_range()
}

///|
pub fn DeepSeekDecoderStage::text_boundary(
  self : Self,
) -> @precision.TextBoundary {
  self.range.text_boundary()
}

///|
pub fn DeepSeekDecoderStage::source_bytes(self : Self) -> Bytes raise {
  let prefix = match self.ingress {
    Some(ingress) => ingress.source_bytes() + b"\n"
    None => b""
  }
  let suffix = match self.egress {
    Some(egress) => b"\n" + egress.source_bytes()
    None => b""
  }
  prefix + self.decoder.source_bytes() + suffix
}

///|
/// All owned weights/state/workspace/residuals/token tables. Borrowed ports,
/// index offsets and transport buffers must be budgeted separately by the caller.
pub fn DeepSeekDecoderStage::required_device_bytes(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
) -> Int64 raise {
  let ingress_bytes = match self.ingress {
    Some(ingress) => ingress.required_device_bytes(checkpoint)
    None => 0L
  }
  let egress_bytes = match self.egress {
    Some(egress) => egress.required_device_bytes(checkpoint)
    None => 0L
  }
  let tables = self.bind_token_tables(checkpoint)
  let table_bytes = tables.fold(init=0L, (total, table) => {
    total + table.i32_bytes()
  })
  self.decoder.required_device_bytes(checkpoint) +
  ingress_bytes +
  egress_bytes +
  2L * self.residual_bytes +
  table_bytes
}

///|
fn DeepSeekDecoderStage::bind_token_tables(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
) -> Array[@materialize.CheckpointI64Table] raise {
  let (first_layer, end_layer) = self.range.layer_range()
  @expert.bind_token_hash_tables(
    checkpoint,
    self.spec,
    first_layer~,
    end_layer~,
  )
}

///|
/// Resolve checkpoint and borrowed inputs before the first allocation. All text
/// and decoder effects join one queue; intermediate residuals never alias.
pub fn DeepSeekDecoderStage::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  stream : @device.Stream,
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  ports : @serial.SerialDecoderWorker,
  index_offsets? : @device.Allocation,
  maximum_device_bytes~ : Int64,
  maximum_host_bytes~ : Int64,
  chunk_bytes~ : Int,
  limits~ : @streaming.StreamingSafetensorsLimits,
) -> Unit raise {
  guard self.state is Fresh else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let (rows, history, vocabulary) = ports.geometry()
  guard rows == self.maximum_rows &&
    history <= self.maximum_context &&
    vocabulary == self.vocabulary &&
    (!self.needs_index || index_offsets is Some(_)) &&
    chunk_bytes > 0 else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let required = self.required_device_bytes(checkpoint)
  guard required <= maximum_device_bytes else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  let tables = self.bind_token_tables(checkpoint)
  let host_bytes = if tables.is_empty() {
    chunk_bytes.to_int64()
  } else {
    @upload.i64_table_host_bytes(tables, chunk_bytes~)
  }
  guard tables.length() == self.token_table_count &&
    host_bytes <= maximum_host_bytes else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  let counts = ports.input(Counts)
  let positions = ports.input(Positions)
  let reset_flag = ports.input(SparseReset)
  let token_ids = ports.input(TokenIds)
  let output_count = ports.input(OutputCount)
  let selected_rows = ports.input(SelectedRows)
  let sampled_tokens = ports.sampled_tokens()
  self.state = Failed
  let token_tables : ArrayView[@device.Allocation] = if tables.is_empty() {
    []
  } else {
    self.token_tables.upload(
      context,
      checkpoint,
      root,
      tables,
      minimum_value=0,
      exclusive_maximum_value=self.spec.routed_expert_count(),
      maximum_device_bytes=tables.fold(init=0L, (total, table) => {
        total + table.i32_bytes()
      }),
      maximum_host_bytes~,
      chunk_bytes~,
      limits~,
    )
    self.token_tables.allocations()
  }
  self.residuals.push(context.allocate(self.residual_bytes))
  self.residuals.push(context.allocate(self.residual_bytes))
  let prefix_launches = match self.ingress {
    Some(ingress) => {
      ingress.prepare(
        context,
        aot,
        checkpoint,
        root,
        maximum_device_bytes=ingress.required_device_bytes(checkpoint),
        chunk_bytes~,
        limits~,
        counts~,
        token_ids~,
        input_residual=self.residuals[0],
      )
      ingress.prefix()
    }
    None => []
  }
  let suffix_launches = match self.egress {
    Some(egress) => {
      egress.prepare(
        context,
        aot,
        checkpoint,
        root,
        maximum_device_bytes=egress.required_device_bytes(checkpoint),
        chunk_bytes~,
        limits~,
        counts~,
        output_count~,
        selected_rows~,
        sampled_tokens~,
        output_residual=self.residuals[1],
      )
      egress.suffix()
    }
    None => []
  }
  self.decoder.prepare(
    context,
    aot,
    stream,
    checkpoint,
    root,
    counts~,
    positions~,
    reset_flag~,
    token_ids~,
    input=self.residuals[0],
    output=self.residuals[1],
    index_offsets?,
    token_tables~,
    prefix_launches~,
    suffix_launches~,
    maximum_device_bytes=self.decoder.required_device_bytes(checkpoint),
    chunk_bytes~,
    limits~,
  )
  self.counts = Some(counts)
  self.state = Prepared
}

///|
pub fn DeepSeekDecoderStage::input_residual(
  self : Self,
) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.residuals[0]
}

///|
pub fn DeepSeekDecoderStage::output_residual(
  self : Self,
) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.residuals[1]
}

///|
pub fn DeepSeekDecoderStage::launch_count(self : Self) -> Int {
  self.decoder.launch_count()
}

///|
pub fn DeepSeekDecoderStage::append_descriptors(
  self : Self,
) -> ArrayView[@device.Allocation] {
  self.decoder.append_descriptors()
}

///|
pub fn DeepSeekDecoderStage::submit(
  self : Self,
  live_rows~ : Int,
) -> Unit raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::Busy(OrderedExecutorEnqueue)
  }
  self.state = Failed
  self.decoder.submit(live_rows~)
  self.state = Active
}

///|
pub fn DeepSeekDecoderStage::poll(self : Self) -> Bool raise {
  guard self.state is Active else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  self.state = Failed
  if self.decoder.poll() {
    self.state = Prepared
    true
  } else {
    self.state = Active
    false
  }
}

///|
/// Only a complete stage may publish whole-model tokens. Partial ranges expose
/// residuals but never independently pretend to be a complete model.
pub fn DeepSeekDecoderStage::bind_worker(
  self : Self,
  ports : @serial.SerialDecoderWorker,
  context : @device.Context,
) -> Unit raise {
  guard self.state is Prepared &&
    self.range.text_boundary() is Complete &&
    self.counts is Some(counts) &&
    counts.same_storage(ports.input(Counts)) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  ports.bind_executor(
    context,
    submit=rows => self.submit(live_rows=rows),
    poll=() => self.poll(),
    close=() => self.close(),
    append_descriptors=self.append_descriptors(),
  )
}

///|
/// Retire worker/transport leases before closing the queue, then weights/state
/// and finally its borrowed residual boundaries.
pub fn DeepSeekDecoderStage::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.decoder.close()
  self.token_tables.close()
  self.counts = None
  if self.egress is Some(egress) {
    egress.close()
  }
  if self.ingress is Some(ingress) {
    ingress.close()
  }
  while self.residuals.last() is Some(buffer) {
    buffer.close()
    ignore(self.residuals.pop())
  }
  self.state = Closed
}
