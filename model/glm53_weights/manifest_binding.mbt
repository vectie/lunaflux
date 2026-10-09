///|
fn same_shape(
  left : ReadOnlyArray[Int64],
  right : ReadOnlyArray[Int64],
) -> Bool {
  if left.length() != right.length() {
    return false
  }
  for index, dimension in left {
    if right[index] != dimension {
      return false
    }
  }
  true
}

///|
/// Validates a family-owned decoded safetensors header in canonical order.
/// Duplicate, reordered, missing, unknown, dtype-shifted, and shape-shifted
/// entries all fail before any payload is materialized.
pub fn bind_observed_tensor_manifest(
  manifest : Glm53TensorManifest,
  observed : ArrayView[Glm53ObservedTensor],
) -> Glm53WeightBinding raise Glm53WeightsError {
  if observed.length() != manifest.tensors.length() {
    raise TensorCountMismatch(
      expected=manifest.tensors.length(),
      actual=observed.length(),
    )
  }
  let names : Map[String, Int] = Map([])
  for ordinal, actual in observed {
    if names.contains(actual.name) {
      raise DuplicateTensor(ordinal~)
    }
    names[actual.name] = ordinal
    let expected = manifest.tensors[ordinal]
    if actual.name != expected.name {
      raise TensorNameMismatch(ordinal~)
    }
    if actual.dtype != expected.dtype {
      raise TensorDTypeMismatch(ordinal~)
    }
    if !same_shape(actual.shape, expected.shape) {
      raise TensorShapeMismatch(ordinal~)
    }
  }
  {
    execution_identity: manifest.execution_identity,
    layout_digest: manifest.layout_digest,
    tensor_count: observed.length(),
  }
}

///|
/// Binds through the shared v1 safetensors metadata only when every tensor is
/// BF16. Official GLM manifests contain required F32 state/router tensors, so
/// the current BF16-only materializer fails closed instead of silently casting.
pub fn bind_safetensors_manifest(
  manifest : Glm53TensorManifest,
  metadata : @safetensors.SafetensorsMetadata,
) -> Glm53WeightBinding raise Glm53WeightsError {
  if metadata.tensor_count() != manifest.tensors.length() {
    raise TensorCountMismatch(
      expected=manifest.tensors.length(),
      actual=metadata.tensor_count(),
    )
  }
  let observed = []
  for ordinal, expected in manifest.tensors {
    if expected.dtype != BFloat16 {
      raise UnsupportedSafetensorsDType(ordinal~)
    }
    guard metadata.tensor(ordinal) is Some(actual) else {
      raise TensorCountMismatch(
        expected=manifest.tensors.length(),
        actual=ordinal,
      )
    }
    observed.push(
      Glm53ObservedTensor::new(
        name=actual.name(),
        dtype=BFloat16,
        shape=actual.shape(),
      ),
    )
  }
  bind_observed_tensor_manifest(manifest, observed)
}
