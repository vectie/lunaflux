///|
priv enum RecurrentLayerState {
  Fresh
  Ready
  Active
  Failed
  Closed
}

///|
/// Executable checkpoint-backed KDA branch, optionally with the complete
/// residual-stream attention envelope and early-layer dense FFN. Owns weights, request state and
/// intermediates, borrowing the module, stream and frame I/O.
pub struct GlmRecurrentLayer {
  priv spec : @glm53_spec.Glm53FlashSpec
  priv layer_index : Int
  priv program : @recurrent.RecurrentDeltaProgram
  priv weights : @upload.CheckpointDeviceWeights
  priv hyper : @hyper.HyperConnectionFrame?
  priv ffn : GlmDenseFfn?
  priv mut status : RecurrentLayerState
}

///|
pub fn GlmRecurrentLayer::new(
  spec : @glm53_spec.Glm53FlashSpec,
  layer_index~ : Int,
  maximum_rows~ : Int,
  maximum_sequences~ : Int,
  cache_slots~ : Int,
  maximum_state_bytes~ : Int64,
) -> Self raise {
  let plan = recurrent_block_plan(
    spec,
    layer_index~,
    maximum_rows~,
    maximum_sequences~,
    cache_slots~,
    maximum_state_bytes~,
  )
  let source = @block.recurrent_block_source(
    plan,
    symbol_prefix="glm_recurrent_" + layer_index.to_string(),
  )
  {
    spec,
    layer_index,
    program: @recurrent.RecurrentDeltaProgram::with_block(source),
    weights: @upload.CheckpointDeviceWeights::new(),
    hyper: None,
    ffn: None,
    status: Fresh,
  }
}

///|
pub fn GlmRecurrentLayer::source_bytes(self : Self) -> Bytes {
  let attention = match self.hyper {
    None => self.program.source_bytes()
    Some(hyper) => hyper.source_bytes() + b"\n" + self.program.source_bytes()
  }
  match self.ffn {
    None => attention
    Some(ffn) => attention + b"\n" + ffn.source_bytes()
  }
}

///|
pub fn GlmRecurrentLayer::workspace_bytes(self : Self) -> Int64 {
  let extra = match self.hyper {
    None => 0L
    Some(hyper) => hyper.workspace_bytes()
  }
  let ffn_bytes = match self.ffn {
    None => 0L
    Some(ffn) => ffn.workspace_bytes()
  }
  self.program.workspace_bytes() + extra + ffn_bytes
}

///|
pub fn GlmRecurrentLayer::device_bytes(self : Self) -> Int64 {
  self.workspace_bytes() + self.weights.device_bytes()
}

///|
/// Bind once and transfer directly into owned device weights. The total budget
/// includes request state/intermediates, not just checkpoint weight bytes.
pub fn GlmRecurrentLayer::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  stream : @device.Stream,
  checkpoint : @precision_materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  maximum_device_bytes~ : Int64,
  chunk_bytes~ : Int,
  limits~ : @streaming_safetensors.StreamingSafetensorsLimits,
  counts~ : @device.Allocation,
  sequence_offsets~ : @device.Allocation,
  sequence_slots~ : @device.Allocation,
  reset_flags~ : @device.Allocation,
  hidden~ : @device.Allocation,
  output~ : @device.Allocation,
) -> Unit raise {
  guard self.status is Fresh && maximum_device_bytes >= self.workspace_bytes() else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.status = Failed
  errdefer {
    self.status = Failed
  }
  let bound = bind_recurrent_weights(
    checkpoint,
    self.spec,
    layer_index=self.layer_index,
  )
  if self.hyper is Some(_) {
    for
      weight in bind_attention_hyper_weights(
        checkpoint,
        self.spec,
        self.layer_index,
      ) {
      bound.push(weight)
    }
  }
  if self.ffn is Some(_) {
    for
      weight in bind_dense_ffn_weights(checkpoint, self.spec, self.layer_index) {
      bound.push(weight)
    }
  }
  self.weights.upload(
    context,
    checkpoint,
    root,
    bound,
    maximum_device_bytes=maximum_device_bytes - self.workspace_bytes(),
    chunk_bytes~,
    limits~,
  )
  let attention_output = match self.ffn {
    None => output
    Some(ffn) => {
      ffn.prepare(context, aot, self.weights, counts~, output~)
      ffn.input()
    }
  }
  let (input, branch_output, prefix, attention_suffix) : (
    @device.Allocation,
    @device.Allocation,
    ArrayView[@device.OrderedKernelLaunch],
    ArrayView[@device.OrderedKernelLaunch],
  ) = match self.hyper {
    None => (hidden, attention_output, [], [])
    Some(hyper) => {
      hyper.prepare(
        context,
        aot,
        counts~,
        input=hidden,
        output=attention_output,
        function_weight=self.weights.allocation(15),
        base=self.weights.allocation(16),
        scale=self.weights.allocation(17),
        norm_weight=self.weights.allocation(18),
        maximum_workspace_bytes=hyper.workspace_bytes(),
      )
      (
        hyper.branch_input(),
        hyper.branch_output(),
        hyper.prefix(),
        hyper.suffix(),
      )
    }
  }
  let suffix = attention_suffix.to_owned()
  if self.ffn is Some(ffn) {
    for launch in ffn.launches {
      suffix.push(launch)
    }
  }
  self.program.prepare_block(
    context,
    aot,
    stream,
    recurrent_device_weights(self.weights),
    maximum_state_bytes=self.program.state_bytes(),
    maximum_workspace_bytes=self.program.workspace_bytes(),
    counts~,
    sequence_offsets~,
    sequence_slots~,
    reset_flags~,
    hidden=input,
    output=branch_output,
    prefix~,
    suffix~,
  )
  self.status = Ready
}

///|
pub fn GlmRecurrentLayer::submit(
  self : Self,
  live_rows~ : Int,
) -> Unit raise @device.DeviceError {
  guard self.status is Ready else {
    raise @device.DeviceError::Busy(OrderedExecutorEnqueue)
  }
  errdefer {
    self.status = Failed
  }
  self.program.submit(live_rows~)
  self.status = Active
}

///|
pub fn GlmRecurrentLayer::poll(self : Self) -> Bool raise @device.DeviceError {
  guard self.status is Active else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  errdefer {
    self.status = Failed
  }
  if self.program.poll() {
    self.status = Ready
    true
  } else {
    false
  }
}

///|
/// Close/abort the queue before releasing its borrowed weights. Failed startup
/// retains explicit cleanup authority, including partially transferred weights.
pub fn GlmRecurrentLayer::close(self : Self) -> Unit raise @device.DeviceError {
  match self.status {
    Closed => return
    Fresh | Ready | Active | Failed => ()
  }
  self.status = Failed
  self.program.abort()
  if self.ffn is Some(ffn) {
    ffn.close()
  }
  if self.hyper is Some(hyper) {
    hyper.close()
  }
  self.weights.close()
  self.status = Closed
}
