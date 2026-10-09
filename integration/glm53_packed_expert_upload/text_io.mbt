///|
/// GLM's final residual collapse is an unweighted mean, not DeepSeek's learned
/// head. Map this model semantic to the generic text ingress/egress plan.
pub fn flash_text_io_plan(
  spec : @glm53_spec.Glm53FlashSpec,
  maximum_rows~ : Int,
  maximum_output_rows~ : Int,
  maximum_workspace_bytes~ : Int64,
  boundary? : @precision_ir.TextBoundary = Complete,
) -> @precision_ir.TextIoPrecision raise {
  @precision_ir.TextIoPrecision::new(
    vocabulary_size=spec.vocabulary_size(),
    hidden_width=spec.hidden_size(),
    stream_count=spec.hyper_connection_streams(),
    maximum_rows~,
    maximum_output_rows~,
    normalization_epsilon=spec.rms_norm_epsilon(),
    maximum_workspace_bytes~,
    boundary~,
  )
}

///|
pub fn bind_text_io_weights(
  checkpoint : @precision_materialize.PackedCheckpoint,
  spec : @glm53_spec.Glm53FlashSpec,
  boundary? : @precision_ir.TextBoundary = Complete,
) -> Array[@precision_materialize.PackedCheckpointWeight] raise {
  let table = @precision_format.Representation::new(
    format=Bf16,
    rows=spec.vocabulary_size(),
    columns=spec.hidden_size(),
  )
  let weights = []
  if boundary.has_ingress() {
    weights.push(
      checkpoint.bind_weight(
        representation=table,
        payload_name="model.language_model.embed_tokens.weight",
      ),
    )
  }
  if boundary.has_egress() {
    weights.push(
      checkpoint.bind_vector(
        representation=@precision_format.Representation::new(
          format=Bf16,
          rows=1,
          columns=spec.hidden_size(),
        ),
        payload_name="model.language_model.norm.weight",
      ),
    )
    weights.push(
      checkpoint.bind_weight(
        representation=table,
        payload_name="lm_head.weight",
      ),
    )
  }
  weights
}

///|
/// Owns streamed global weights and a generic prepared frame. Add prefix/suffix
/// to the decoder queue; close that queue before closing this owner.
pub struct GlmTextIo {
  priv spec : @glm53_spec.Glm53FlashSpec
  priv boundary : @precision_ir.TextBoundary
  priv frame : @text_frame.TextIoFrame
  priv weights : @upload.CheckpointDeviceWeights
  priv mut status : RecurrentLayerState
}

///|
pub fn GlmTextIo::new(
  spec : @glm53_spec.Glm53FlashSpec,
  maximum_rows~ : Int,
  maximum_output_rows~ : Int,
  maximum_workspace_bytes~ : Int64,
  boundary? : @precision_ir.TextBoundary = Complete,
) -> Self raise {
  let plan = flash_text_io_plan(
    spec,
    maximum_rows~,
    maximum_output_rows~,
    maximum_workspace_bytes~,
    boundary~,
  )
  {
    spec,
    boundary,
    frame: @text_frame.TextIoFrame::new(
      @text_source.text_io_source(plan, symbol_prefix="glm_text_io"),
      boundary~,
    ),
    weights: @upload.CheckpointDeviceWeights::new(),
    status: Fresh,
  }
}

///|
pub fn GlmTextIo::source_bytes(self : Self) -> Bytes {
  self.frame.source_bytes()
}

///|
pub fn GlmTextIo::required_device_bytes(
  self : Self,
  checkpoint : @precision_materialize.PackedCheckpoint,
) -> Int64 raise {
  bind_text_io_weights(checkpoint, self.spec, boundary=self.boundary).fold(
    init=self.frame.workspace_bytes(),
    (bytes, weight) => bytes + weight.buffer_bytes(),
  )
}

///|
pub fn GlmTextIo::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  checkpoint : @precision_materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  maximum_device_bytes~ : Int64,
  chunk_bytes~ : Int,
  limits~ : @streaming_safetensors.StreamingSafetensorsLimits,
  counts~ : @device.Allocation,
  token_ids~ : @device.Allocation,
  output_count~ : @device.Allocation,
  selected_rows~ : @device.Allocation,
  input_residual~ : @device.Allocation,
  output_residual~ : @device.Allocation,
  sampled_tokens~ : @device.Allocation,
) -> Unit raise {
  guard self.status is Fresh &&
    self.required_device_bytes(checkpoint) <= maximum_device_bytes else {
    raise @precision_ir.PrecisionPlanError::MemoryBudgetExceeded
  }
  self.status = Failed
  if self.boundary is Interior {
    self.frame.prepare_interior()
    self.status = Prepared
    return
  }
  self.weights.upload(
    context,
    checkpoint,
    root,
    bind_text_io_weights(checkpoint, self.spec, boundary=self.boundary),
    maximum_device_bytes=maximum_device_bytes - self.frame.workspace_bytes(),
    chunk_bytes~,
    limits~,
  )
  match self.boundary {
    Ingress =>
      self.frame.prepare_ingress(
        aot,
        counts~,
        token_ids~,
        input_residual~,
        embedding=self.weights.allocation(0),
      )
    Egress =>
      self.frame.prepare_egress(
        context,
        aot,
        counts~,
        output_count~,
        selected_rows~,
        output_residual~,
        norm_weight=self.weights.allocation(0),
        head_weight=self.weights.allocation(1),
        sampled_tokens~,
        maximum_workspace_bytes=self.frame.workspace_bytes(),
      )
    Complete =>
      self.frame.prepare(
        context,
        aot,
        counts~,
        token_ids~,
        output_count~,
        selected_rows~,
        input_residual~,
        output_residual~,
        embedding=self.weights.allocation(0),
        norm_weight=self.weights.allocation(1),
        head_weight=self.weights.allocation(2),
        sampled_tokens~,
        maximum_workspace_bytes=self.frame.workspace_bytes(),
      )
    Interior => ()
  }
  self.status = Prepared
}

///|
pub fn GlmTextIo::prefix(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  self.frame.prefix()
}

///|
pub fn GlmTextIo::suffix(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  self.frame.suffix()
}

///|
pub fn GlmTextIo::logits(self : Self) -> @device.Allocation raise {
  self.frame.logits()
}

///|
pub fn GlmTextIo::close(self : Self) -> Unit raise {
  if self.status is Closed {
    return
  }
  self.status = Failed
  self.frame.close()
  self.weights.close()
  self.status = Closed
}
