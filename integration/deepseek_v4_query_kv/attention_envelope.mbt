///|
priv enum AttentionBranch {
  Window(DeepSeekWindowSublayer)
  Compressed(DeepSeekCompressedSublayer)
}

///|
/// Complete attention residual-stream envelope. All effects join the enclosing
/// decoder queue; the learned output writes into the envelope's branch buffer.
pub struct DeepSeekAttentionEnvelope {
  priv hyper : @hyper.DeepSeekHyperConnection
  priv branch : AttentionBranch
  priv launches : Array[@device.OrderedKernelLaunch]
  priv descriptors : Array[@device.Allocation]
  priv needs_index : Bool
  priv mut state : State
}

///|
pub fn DeepSeekAttentionEnvelope::new(
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  maximum_rows~ : Int,
  maximum_context~ : Int,
  maximum_state_bytes~ : Int64,
  maximum_workspace_bytes~ : Int64,
) -> Self raise {
  let hyper = @hyper.DeepSeekHyperConnection::new(
    spec,
    layer~,
    branch=Attention,
    maximum_rows~,
  )
  let branch = match spec.compression_ratio(layer) {
    Some(0) => {
      ignore(
        window_attention_plan(spec, layer~, maximum_rows~, maximum_state_bytes~),
      )
      Window(
        DeepSeekWindowSublayer::new(
          spec,
          layer~,
          maximum_rows~,
          maximum_workspace_bytes~,
        ),
      )
    }
    Some(4 | 128) =>
      Compressed(
        DeepSeekCompressedSublayer::new(
          spec,
          layer~,
          maximum_rows~,
          maximum_context~,
          maximum_state_bytes~,
          maximum_workspace_bytes~,
        ),
      )
    _ => raise @precision.PrecisionPlanError::IllegalPlacement
  }
  {
    hyper,
    branch,
    launches: [],
    descriptors: [],
    needs_index: spec.compression_ratio(layer) == Some(4),
    state: Fresh,
  }
}

///|
pub fn DeepSeekAttentionEnvelope::source_bytes(self : Self) -> Bytes {
  let branch = match self.branch {
    Window(branch) => branch.source_bytes()
    Compressed(branch) => branch.source_bytes()
  }
  self.hyper.source_bytes() + b"\n" + branch
}

///|
fn AttentionBranch::required_device_bytes(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
) -> Int64 raise {
  match self {
    Window(branch) =>
      branch.required_device_bytes(checkpoint, output_borrowed=true)
    Compressed(branch) =>
      branch.required_device_bytes(checkpoint, output_borrowed=true)
  }
}

///|
pub fn DeepSeekAttentionEnvelope::required_device_bytes(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
) -> Int64 raise {
  self.hyper.required_device_bytes(checkpoint) +
  self.branch.required_device_bytes(checkpoint)
}

///|
pub fn DeepSeekAttentionEnvelope::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  counts~ : @device.Allocation,
  positions~ : @device.Allocation,
  reset_flag~ : @device.Allocation,
  input~ : @device.Allocation,
  output~ : @device.Allocation,
  index_offsets? : @device.Allocation,
  maximum_device_bytes~ : Int64,
  chunk_bytes~ : Int,
  limits~ : @streaming.StreamingSafetensorsLimits,
) -> Unit raise {
  guard self.needs_index == (index_offsets is Some(_)) else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let hyper_bytes = self.hyper.required_device_bytes(checkpoint)
  let branch_bytes = self.branch.required_device_bytes(checkpoint)
  guard self.state is Fresh &&
    hyper_bytes <= maximum_device_bytes &&
    branch_bytes <= maximum_device_bytes - hyper_bytes else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  self.state = Failed
  self.hyper.prepare(
    context,
    aot,
    checkpoint,
    root,
    counts~,
    input~,
    output~,
    maximum_device_bytes=hyper_bytes,
    chunk_bytes~,
    limits~,
  )
  let hidden = self.hyper.branch_input()
  let branch_output = self.hyper.branch_output()
  match self.branch {
    Window(branch) => {
      branch.prepare(
        context,
        aot,
        checkpoint,
        root,
        counts~,
        positions~,
        reset_flag~,
        hidden~,
        output=branch_output,
        maximum_device_bytes=branch_bytes,
        chunk_bytes~,
        limits~,
      )
      self.launches.append(self.hyper.prefix())
      self.launches.append(branch.launches())
      self.descriptors.push(branch.append_descriptor())
    }
    Compressed(branch) => {
      branch.prepare(
        context,
        aot,
        checkpoint,
        root,
        counts~,
        positions~,
        reset_flag~,
        hidden~,
        output=branch_output,
        index_offsets?,
        maximum_device_bytes=branch_bytes,
        chunk_bytes~,
        limits~,
      )
      self.launches.append(self.hyper.prefix())
      self.launches.append(branch.launches())
      self.descriptors.append(branch.append_descriptors())
    }
  }
  self.launches.append(self.hyper.suffix())
  self.state = Prepared
}

///|
pub fn DeepSeekAttentionEnvelope::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}

///|
pub fn DeepSeekAttentionEnvelope::append_descriptors(
  self : Self,
) -> ArrayView[@device.Allocation] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.descriptors[:]
}

///|
pub fn DeepSeekAttentionEnvelope::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.launches.clear()
  self.descriptors.clear()
  match self.branch {
    Window(branch) => branch.close()
    Compressed(branch) => branch.close()
  }
  self.hyper.close()
  self.state = Closed
}
