///|
priv enum SliceState {
  Fresh
  Prepared
  Active
  Failed
  Closed
}

///|
/// Contiguous base-model layers and their compact checkpoint banks. Owns one
/// ordered queue and at most two inter-layer residual frames. Text boundaries,
/// request metadata and startup-narrowed token tables remain caller-owned.
pub struct DeepSeekDecoderSlice {
  priv spec : @spec.DeepSeekV4ModelSpec
  priv first_layer : Int
  priv end_layer : Int
  priv maximum_rows : Int
  priv residual_bytes : Int64
  priv layers : Array[StageLayer]
  priv capture_layers : ReadOnlyArray[Int]
  priv capture : @capture.StreamMeanCaptureFrame?
  priv buffers : Array[@device.Allocation]
  priv descriptors : Array[@device.Allocation]
  priv mut queue : @device.OrderedKernelExecutor?
  priv mut launch_count : Int
  priv mut state : SliceState
}

///|
pub fn DeepSeekDecoderSlice::new(
  spec : @spec.DeepSeekV4ModelSpec,
  first_layer~ : Int,
  end_layer~ : Int,
  maximum_rows~ : Int,
  maximum_context~ : Int,
  maximum_layer_state_bytes~ : Int64,
  maximum_layer_workspace_bytes~ : Int64,
  maximum_layer_expert_bytes~ : Int64,
  accuracy~ : @precision.Accuracy,
  capture_prediction_inputs? : Bool = false,
) -> Self raise {
  guard first_layer >= 0 &&
    end_layer > first_layer &&
    end_layer <= spec.layer_count() &&
    maximum_rows > 0 &&
    maximum_context >= maximum_rows &&
    maximum_context <= spec.max_position_embeddings() else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  guard !capture_prediction_inputs || spec.has_dspark() else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let capture_layers = if capture_prediction_inputs {
    spec
    .dspark_target_layer_ids()
    .filter(layer => layer >= first_layer && layer < end_layer)
  } else {
    []
  }
  let capture = if capture_layers.is_empty() {
    None
  } else {
    let plan = @precision.StreamMeanCapturePrecision::new(
      stream_count=spec.hyper_connection_width(),
      hidden_width=spec.hidden_size(),
      maximum_rows~,
      segment_count=capture_layers.length(),
      maximum_output_bytes=maximum_layer_workspace_bytes,
    )
    Some(
      @capture.StreamMeanCaptureFrame::new(
        @capture_source.stream_mean_capture_source(
          plan,
          symbol="deepseek_prediction_capture_" + first_layer.to_string(),
        ),
      ),
    )
  }
  let layers = Array::makei(end_layer - first_layer, i => {
    let index = first_layer + i
    {
      index,
      block: DeepSeekDecoderBlock::new(
        spec,
        layer=index,
        maximum_rows~,
        maximum_context~,
        maximum_state_bytes=maximum_layer_state_bytes,
        maximum_workspace_bytes=maximum_layer_workspace_bytes,
        maximum_expert_bytes=maximum_layer_expert_bytes,
        accuracy~,
      ),
      routed: @expert.expert_compute_plan(
        spec,
        layer_index=index,
        expert_indices=Array::makei(spec.routed_expert_count(), i => i),
        maximum_rows~,
        memory_ceiling=maximum_layer_expert_bytes,
        accuracy~,
      ),
      shared: @expert.shared_expert_compute_plan(
        spec,
        layer_index=index,
        maximum_rows~,
        memory_ceiling=maximum_layer_expert_bytes,
        accuracy~,
      ),
      router: @upload.CheckpointDeviceWeights::new(),
      banks: [],
    }
  })
  {
    spec,
    first_layer,
    end_layer,
    maximum_rows,
    residual_bytes: maximum_rows.to_int64() *
    spec.hidden_size().to_int64() *
    spec.hyper_connection_width().to_int64() *
    2L,
    layers,
    capture_layers: ReadOnlyArray::from_array(capture_layers),
    capture,
    buffers: [],
    descriptors: [],
    queue: None,
    launch_count: 0,
    state: Fresh,
  }
}

///|
pub fn DeepSeekDecoderSlice::layer_range(self : Self) -> (Int, Int) {
  (self.first_layer, self.end_layer)
}

///|
pub fn DeepSeekDecoderSlice::source_bytes(self : Self) -> Bytes raise {
  let source = self.layers.fold(init=b"", (source, layer) => {
    source + b"\n" + layer.block.source_bytes()
  })
  match self.capture {
    Some(frame) => source + b"\n" + frame.source_bytes()
    None => source
  }
}

///|
/// Compact expert-bank portion, without private frames or borrowed metadata.
pub fn DeepSeekDecoderSlice::expert_bank_bytes(self : Self) -> Int64 {
  self.layers.fold(init=0L, (bytes, layer) => bytes + layer.bank_bytes())
}

///|
/// Complete owned stage bytes: all weights/banks, persistent attention state,
/// private frames and reusable residuals. Add caller text/ports/tables once.
pub fn DeepSeekDecoderSlice::required_device_bytes(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
) -> Int64 raise {
  self.layers.fold(
    init=self.residual_bytes * (self.layers.length() - 1).min(2).to_int64() +
      self.prediction_capture_bytes(),
    (bytes, layer) => bytes + layer.required_device_bytes(self.spec, checkpoint),
  )
}

///|
pub fn DeepSeekDecoderSlice::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  stream : @device.Stream,
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  counts~ : @device.Allocation,
  positions~ : @device.Allocation,
  reset_flag~ : @device.Allocation,
  input~ : @device.Allocation,
  output~ : @device.Allocation,
  token_ids? : @device.Allocation,
  token_tables? : ArrayView[@device.Allocation] = [],
  maximum_device_bytes~ : Int64,
  chunk_bytes~ : Int,
  limits~ : @streaming.StreamingSafetensorsLimits,
  prefix_launches? : ArrayView[@device.OrderedKernelLaunch] = [],
  suffix_launches? : ArrayView[@device.OrderedKernelLaunch] = [],
) -> Unit raise {
  guard self.state is Fresh else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let hash_count = (self.end_layer.min(self.spec.hash_layer_count()) -
  self.first_layer).max(0)
  guard token_tables.length() == hash_count &&
    (hash_count == 0 || token_ids is Some(_)) &&
    chunk_bytes > 0 else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  guard self.required_device_bytes(checkpoint) <= maximum_device_bytes else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  for port in [input, output] {
    context.validate_allocation_region(
      port,
      offset=0L,
      byte_count=self.residual_bytes,
      alignment=2L,
    )
  }
  self.state = Failed
  for _ in 0..<(self.layers.length() - 1).min(2) {
    self.buffers.push(context.allocate(self.residual_bytes))
  }
  if self.capture is Some(frame) {
    frame.prepare(
      context,
      aot,
      counts~,
      maximum_device_bytes=self.prediction_capture_bytes(),
    )
  }
  let launches = prefix_launches.to_owned()
  // The file session exists only during startup; no filesystem effect enters
  // submit/poll. Only this rank's contiguous layer range is uploaded.
  checkpoint.with_transfer_session(root, limits, checkpoint => {
    for index, layer in self.layers {
      layer.upload(self.spec, context, checkpoint, root, chunk_bytes~, limits~)
      let predecessor = if index == 0 {
        input
      } else {
        self.buffers[(index - 1) % 2]
      }
      let successor = if index + 1 == self.layers.length() {
        output
      } else {
        self.buffers[index % 2]
      }
      let table = if layer.index < self.spec.hash_layer_count() {
        Some(token_tables[index])
      } else {
        None
      }
      layer.block.prepare(
        context,
        aot,
        checkpoint,
        root,
        counts~,
        positions~,
        reset_flag~,
        input=predecessor,
        output=successor,
        routed_bank=layer.banks[0],
        shared_bank=layer.banks[1],
        router_weight=layer.router.allocation(0),
        correction_bias?=if layer.index >= self.spec.hash_layer_count() {
          Some(layer.router.allocation(1))
        } else {
          None
        },
        token_ids?=if table is Some(_) { token_ids } else { None },
        token_table?=table,
        token_table_offset?=if table is Some(_) { Some(0L) } else { None },
        maximum_device_bytes=layer.block.required_device_bytes(checkpoint),
        chunk_bytes~,
        limits~,
      )
      launches.append(layer.block.launches())
      self.descriptors.append(layer.block.append_descriptors())
      if self.capture is Some(frame) {
        for segment, target in self.capture_layers {
          if target == layer.index {
            launches.push(frame.capture(successor, segment~))
          }
        }
      }
    }
  })
  launches.append(suffix_launches)
  self.queue = Some(context.create_ordered_kernel_executor(stream, launches))
  self.launch_count = launches.length()
  self.state = Prepared
}

///|
pub fn DeepSeekDecoderSlice::append_descriptors(
  self : Self,
) -> ArrayView[@device.Allocation] {
  self.descriptors[:]
}

///|
pub fn DeepSeekDecoderSlice::launch_count(self : Self) -> Int {
  self.launch_count
}

///|
pub fn DeepSeekDecoderSlice::submit(
  self : Self,
  live_rows~ : Int,
) -> Unit raise @device.DeviceError {
  guard self.state is Prepared && self.queue is Some(queue) else {
    raise @device.DeviceError::Busy(OrderedExecutorEnqueue)
  }
  guard live_rows > 0 && live_rows <= self.maximum_rows else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  self.state = Failed
  for stage in 0..<self.launch_count {
    queue.enqueue(stage)
  }
  queue.record_completion()
  self.state = Active
}

///|
pub fn DeepSeekDecoderSlice::poll(
  self : Self,
) -> Bool raise @device.DeviceError {
  guard self.state is Active && self.queue is Some(queue) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  self.state = Failed
  if queue.poll_completion() {
    queue.reset()
    self.state = Prepared
    true
  } else {
    self.state = Active
    false
  }
}

///|
pub fn DeepSeekDecoderSlice::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  if self.queue is Some(queue) {
    queue.abort()
    queue.close()
    self.queue = None
  }
  for layer in self.layers.rev() {
    layer.close()
  }
  if self.capture is Some(frame) {
    frame.close()
  }
  while self.buffers.last() is Some(buffer) {
    buffer.close()
    ignore(self.buffers.pop())
  }
  self.state = Closed
}
