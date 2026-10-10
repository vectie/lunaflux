///|
/// The first stage replicates the official BF16 embedding over mHC streams.
/// This model semantic lowers through the generic text IR/frame. It owns no
/// vocabulary-head workspace or output weights.
pub fn text_ingress_plan(
  spec : @spec.DeepSeekV4ModelSpec,
  maximum_rows~ : Int,
) -> @precision.TextIoPrecision raise {
  @precision.TextIoPrecision::new(
    vocabulary_size=spec.vocabulary_size(),
    hidden_width=spec.hidden_size(),
    stream_count=spec.hyper_connection_width(),
    maximum_rows~,
    maximum_output_rows=1,
    normalization_epsilon=spec.rms_norm_epsilon(),
    maximum_workspace_bytes=0L,
    boundary=Ingress,
  )
}

///|
pub fn bind_text_ingress_weight(
  checkpoint : @materialize.PackedCheckpoint,
  spec : @spec.DeepSeekV4ModelSpec,
) -> @materialize.PackedCheckpointWeight raise {
  checkpoint.bind_weight(
    representation=@format.Representation::new(
      format=Bf16,
      rows=spec.vocabulary_size(),
      columns=spec.hidden_size(),
    ),
    payload_name="embed.weight",
  )
}

///|
/// Streamed embedding plus a borrowed ingress launch. The parent decoder queue
/// must be retired/closed before this owner releases its weights and function.
pub struct DeepSeekTextIngress {
  priv spec : @spec.DeepSeekV4ModelSpec
  priv frame : @text.TextIoFrame
  priv weights : @upload.CheckpointDeviceWeights
  priv mut state : State
}

///|
pub fn DeepSeekTextIngress::new(
  spec : @spec.DeepSeekV4ModelSpec,
  maximum_rows~ : Int,
) -> Self raise {
  {
    spec,
    frame: @text.TextIoFrame::new(
      @text_source.text_io_source(
        text_ingress_plan(spec, maximum_rows~),
        symbol_prefix="deepseek_text_ingress",
      ),
      boundary=Ingress,
    ),
    weights: @upload.CheckpointDeviceWeights::new(),
    state: Fresh,
  }
}

///|
pub fn DeepSeekTextIngress::source_bytes(self : Self) -> Bytes {
  self.frame.source_bytes()
}

///|
pub fn DeepSeekTextIngress::required_device_bytes(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
) -> Int64 raise {
  bind_text_ingress_weight(checkpoint, self.spec).buffer_bytes()
}

///|
pub fn DeepSeekTextIngress::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  maximum_device_bytes~ : Int64,
  chunk_bytes~ : Int,
  limits~ : @streaming.StreamingSafetensorsLimits,
  counts~ : @device.Allocation,
  token_ids~ : @device.Allocation,
  input_residual~ : @device.Allocation,
) -> Unit raise {
  guard self.state is Fresh &&
    self.required_device_bytes(checkpoint) <= maximum_device_bytes else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  self.state = Failed
  self.weights.upload(
    context,
    checkpoint,
    root,
    [bind_text_ingress_weight(checkpoint, self.spec)],
    maximum_device_bytes~,
    chunk_bytes~,
    limits~,
  )
  self.frame.prepare_ingress(
    aot,
    counts~,
    token_ids~,
    input_residual~,
    embedding=self.weights.allocation(0),
  )
  self.state = Prepared
}

///|
pub fn DeepSeekTextIngress::prefix(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.frame.prefix()
}

///|
pub fn DeepSeekTextIngress::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.frame.close()
  self.weights.close()
  self.state = Closed
}
