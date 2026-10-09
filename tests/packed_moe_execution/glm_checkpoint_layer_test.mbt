///|
/// Test-only common lifecycle for the two distinct attention implementations.
priv enum CheckpointTestLayer {
  Recurrent(@glm53_packed_expert_upload.GlmRecurrentLayer)
  Sparse(@glm53_packed_expert_upload.GlmSparseLayer)
}

///|
fn CheckpointTestLayer::workspace_bytes(self : Self) -> Int64 {
  match self {
    Recurrent(layer) => layer.workspace_bytes()
    Sparse(layer) => layer.workspace_bytes() + layer.state_bytes()
  }
}

///|
fn CheckpointTestLayer::device_bytes(self : Self) -> Int64 {
  match self {
    Recurrent(layer) => layer.device_bytes()
    Sparse(layer) => layer.device_bytes()
  }
}

///|
fn CheckpointTestLayer::source_bytes(self : Self) -> Bytes {
  match self {
    Recurrent(layer) => layer.source_bytes()
    Sparse(layer) => layer.source_bytes()
  }
}

///|
fn CheckpointTestLayer::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  stream : @device.Stream,
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  operand : @device.Allocation,
  maximum_device_bytes : Int64,
  chunk_bytes : Int,
  limits : @streaming.StreamingSafetensorsLimits,
) -> Unit raise {
  match self {
    Recurrent(layer) =>
      layer.prepare(
        context,
        aot,
        stream,
        checkpoint,
        root,
        maximum_device_bytes~,
        chunk_bytes~,
        limits~,
        counts=operand,
        sequence_offsets=operand,
        sequence_slots=operand,
        reset_flags=operand,
        hidden=operand,
        output=operand,
      )
    Sparse(layer) =>
      layer.prepare(
        context,
        aot,
        stream,
        checkpoint,
        root,
        maximum_device_bytes~,
        chunk_bytes~,
        limits~,
        counts=operand,
        reset_flag=operand,
        hidden=operand,
        output=operand,
      )
  }
}

///|
fn CheckpointTestLayer::submit(
  self : Self,
  live_rows~ : Int,
) -> Unit raise @device.DeviceError {
  match self {
    Recurrent(layer) => layer.submit(live_rows~)
    Sparse(layer) => layer.submit(live_rows~)
  }
}

///|
fn CheckpointTestLayer::poll(self : Self) -> Bool raise @device.DeviceError {
  match self {
    Recurrent(layer) => layer.poll()
    Sparse(layer) => layer.poll()
  }
}

///|
fn CheckpointTestLayer::close(self : Self) -> Unit raise @device.DeviceError {
  match self {
    Recurrent(layer) => layer.close()
    Sparse(layer) => layer.close()
  }
}
