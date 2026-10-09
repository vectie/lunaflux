///|
fn GlmHybridReferenceLimits::check_sequence(
  self : Self,
  sequence_length : Int,
) -> Unit raise GlmHybridReferenceError {
  if sequence_length <= 0 || sequence_length > self.maximum_sequence_length {
    raise BoundsExceeded
  }
}

///|
fn GlmHybridReferenceLimits::check_heads(
  self : Self,
  head_count : Int,
  head_dimension : Int,
) -> Unit raise GlmHybridReferenceError {
  if head_count <= 0 ||
    head_count > self.maximum_head_count ||
    head_dimension <= 0 ||
    head_dimension > self.maximum_head_dimension {
    raise BoundsExceeded
  }
}

///|
fn GlmHybridReferenceLimits::check_channels(
  self : Self,
  channel_count : Int,
) -> Unit raise GlmHybridReferenceError {
  if channel_count <= 0 || channel_count > self.maximum_channel_count {
    raise BoundsExceeded
  }
}

///|
fn GlmHybridReferenceLimits::check_vision_patches(
  self : Self,
  patch_count : Int,
) -> Unit raise GlmHybridReferenceError {
  if patch_count <= 0 || patch_count > self.maximum_vision_patch_count {
    raise BoundsExceeded
  }
}

///|
fn finite(value : Double) -> Bool {
  !value.is_nan() && !value.is_inf()
}

///|
fn require_finite(
  values : ArrayView[Double],
) -> Unit raise GlmHybridReferenceError {
  for value in values {
    if !finite(value) {
      raise NonFiniteInput
    }
  }
}
