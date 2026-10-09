///|
/// Model adaptation is resolved at startup. The warmed executor only sees the
/// resulting ordered launches, not attention-family or feed-forward dispatch.
priv enum GlmDecoderLayer {
  Recurrent(GlmRecurrentLayer)
  Sparse(GlmSparseLayer)
}

///|
fn GlmDecoderLayer::source_bytes(self : Self) -> Bytes {
  match self {
    Recurrent(layer) => layer.source_bytes()
    Sparse(layer) => layer.source_bytes()
  }
}

///|
fn GlmDecoderLayer::workspace_bytes(self : Self) -> Int64 {
  match self {
    Recurrent(layer) => layer.workspace_bytes()
    Sparse(layer) => layer.workspace_bytes() + layer.state_bytes()
  }
}

///|
fn GlmDecoderLayer::state_bytes(self : Self) -> Int64 {
  match self {
    Recurrent(layer) => layer.program.state_bytes()
    Sparse(layer) => layer.state_bytes()
  }
}

///|
fn GlmDecoderLayer::device_bytes(self : Self) -> Int64 {
  match self {
    Recurrent(layer) => layer.device_bytes()
    Sparse(layer) => layer.device_bytes()
  }
}

///|
fn GlmDecoderLayer::required_device_bytes(
  self : Self,
  checkpoint : @precision_materialize.PackedCheckpoint,
) -> Int64 raise {
  match self {
    Recurrent(layer) => layer.required_device_bytes(checkpoint)
    Sparse(layer) => layer.required_device_bytes(checkpoint)
  }
}

///|
fn GlmDecoderLayer::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  checkpoint : @precision_materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  maximum_device_bytes~ : Int64,
  chunk_bytes~ : Int,
  limits~ : @streaming_safetensors.StreamingSafetensorsLimits,
  counts~ : @device.Allocation,
  sequence_offsets~ : @device.Allocation,
  sequence_slots~ : @device.Allocation,
  reset_flags~ : @device.Allocation,
  sparse_reset_flag~ : @device.Allocation,
  hidden~ : @device.Allocation,
  output~ : @device.Allocation,
) -> Unit raise {
  match self {
    Recurrent(layer) =>
      layer.prepare_frame(
        context,
        aot,
        checkpoint,
        root,
        maximum_device_bytes~,
        chunk_bytes~,
        limits~,
        counts~,
        sequence_offsets~,
        sequence_slots~,
        reset_flags~,
        hidden~,
        output~,
      )
    Sparse(layer) =>
      layer.prepare_frame(
        context,
        aot,
        checkpoint,
        root,
        maximum_device_bytes~,
        chunk_bytes~,
        limits~,
        counts~,
        reset_flag=sparse_reset_flag,
        hidden~,
        output~,
      )
  }
}

///|
fn GlmDecoderLayer::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise @device.DeviceError {
  match self {
    Recurrent(layer) => layer.launches()
    Sparse(layer) => layer.launches()
  }
}

///|
fn GlmDecoderLayer::close(self : Self) -> Unit raise @device.DeviceError {
  match self {
    Recurrent(layer) => layer.close()
    Sparse(layer) => layer.close()
  }
}
