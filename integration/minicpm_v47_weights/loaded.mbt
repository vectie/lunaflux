///|
priv enum LoadState {
  Fresh
  Failed
  Prepared
  Closed
}

///|
/// Owns final device weight allocations. Root/checkpoint metadata/context and
/// transfer scratch are borrowed. Close forward frames/executors before Loaded.
pub struct Loaded {
  priv plan : Plan
  priv buffers : Array[@device.Allocation]
  priv layers : Array[@decoder.LayerWeights]
  priv mut text : @text.Weights?
  priv mut state : LoadState
}

///|
pub fn Loaded::new(plan : Plan) -> Self {
  { plan, buffers: [], layers: [], text: None, state: Fresh, }
}

///|
/// Prepare once. Admission precedes allocation; copy into final buffers using
/// one bounded scratch and the common transfer session. Close partial failures.
pub fn Loaded::prepare(
  self : Self,
  context : @device.Context,
  root : @approved_fs.ApprovedRoot,
  maximum_device_bytes~ : Int64,
  scratch~ : FixedArray[Byte],
  limits~ : @streaming.StreamingSafetensorsLimits,
) -> Unit raise {
  guard self.state is Fresh &&
    maximum_device_bytes >= self.plan.bytes &&
    scratch.length() > 0 else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  self.state = Failed
  for entry in self.plan.allocations {
    self.buffers.push(context.allocate(entry.bytes))
  }
  let copy_error : Ref[@device.DeviceError?] = Ref::new(None)
  self.plan.transfer(root, scratch~, limits~, consume=(
    ordinal,
    offset,
    bytes,
    count,
  ) => {
    try
      context.copy_from_fixed_host(
        self.buffers[ordinal],
        bytes,
        0L,
        offset,
        count.to_int64(),
      )
    catch {
      error => {
        copy_error.val = Some(error)
        false
      }
    } noraise {
      _ => true
    }
  }) catch {
    error =>
      match copy_error.val {
        Some(primary) => raise primary
        None => raise error
      }
  }
  for indices in self.plan.layers {
    self.layers.push(bind_layer(indices, self.buffers[:]))
  }
  let indices = self.plan.text
  self.text = Some({
    embedding: self.buffers[indices.embedding],
    final_norm: self.buffers[indices.final_norm],
    language_head: self.buffers[indices.language_head],
  })
  self.state = Prepared
}

///|
pub fn Loaded::layer_weights(
  self : Self,
) -> ArrayView[@decoder.LayerWeights] raise @device.DeviceError {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  self.layers[:]
}

///|
pub fn Loaded::text_weights(
  self : Self,
) -> @text.Weights raise @device.DeviceError {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  self.text.unwrap()
}

///|
pub fn Loaded::close(self : Self) -> Unit raise @device.DeviceError {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.layers.clear()
  self.text = None
  while self.buffers.last() is Some(buffer) {
    buffer.close()
    ignore(self.buffers.pop())
  }
  self.state = Closed
}
