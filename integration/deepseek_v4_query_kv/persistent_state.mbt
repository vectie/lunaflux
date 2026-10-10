///|
pub fn DeepSeekAttentionEnvelope::persistent_state(
  self : Self,
) -> Array[@device.Allocation] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(DeviceCopyCreate)
  }
  match self.branch {
    Window(branch) => branch.attention.window.persistent_state()
    Compressed(branch) => branch.attention.persistent_state()
  }
}

///|
fn DeepSeekCompressedAttention::persistent_state(
  self : Self,
) -> Array[@device.Allocation] raise {
  let state = self.attention.persistent_state()
  state.append(self.cache.persistent_state())
  if self.indexer is Some(indexer) {
    state.append(indexer.cache.persistent_state())
  }
  state
}

///|
fn DeepSeekCompressedKvCache::persistent_state(
  self : Self,
) -> Array[@device.Allocation] raise {
  let state = self.pool.frame.persistent_state()
  state.append(self.cache.persistent_state())
  state
}
