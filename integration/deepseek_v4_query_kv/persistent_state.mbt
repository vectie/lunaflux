///|
/// Pure snapshot placement cost; weights and transient query frames are not
/// replicated. Keep this enumeration paired with persistent_state below.
pub fn DeepSeekAttentionEnvelope::persistent_state_bytes(self : Self) -> Int64 {
  match self.branch {
    Window(branch) => branch.attention.window.persistent_state_bytes()
    Compressed(branch) => branch.attention.persistent_state_bytes()
  }
}

///|
fn DeepSeekCompressedAttention::persistent_state_bytes(self : Self) -> Int64 {
  self.attention.persistent_state_bytes() +
  self.cache.persistent_state_bytes() +
  (match self.indexer {
    Some(indexer) => indexer.cache.persistent_state_bytes()
    None => 0L
  })
}

///|
fn DeepSeekCompressedKvCache::persistent_state_bytes(self : Self) -> Int64 {
  self.pool.frame.persistent_state_bytes() + self.cache.persistent_state_bytes()
}

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
