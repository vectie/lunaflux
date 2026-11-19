///|
/// Joins semantic requirements to shard tensor locations exactly once at
/// preparation. Shards must have passed the common safetensors header reader.
/// No tensor payload is read, hashed, converted, or allocated here.
pub fn bind(
  spec : @model.Spec,
  shards : ArrayView[@safetensors.SafetensorsMetadata],
) -> BoundWeights raise WeightError {
  let required = requirements(spec)
  let locations : Map[String, (Int32, Int32)] = Map([])
  for shard, metadata in shards {
    for tensor in 0..<metadata.tensor_count() {
      let entry = metadata.tensor(tensor).unwrap()
      guard !locations.contains(entry.name()) else {
        raise InvalidWeights("duplicate tensor across shards: " + entry.name())
      }
      locations[entry.name()] = (shard, tensor)
    }
  }
  guard locations.length() == required.length() else {
    raise InvalidWeights("tensor count differs from the selected checkpoint")
  }
  let bindings : Array[Binding] = []
  let indices : Map[String, Int32] = Map([])
  let mut total = 0L
  for requirement in required {
    guard locations.get(requirement.name) is Some((shard, index)) else {
      raise InvalidWeights("missing tensor: " + requirement.name)
    }
    let tensor = shards[shard].tensor(index).unwrap()
    guard tensor.dtype() is BFloat16 && tensor.shape() == requirement.shape else {
      raise InvalidWeights(
        "tensor shape or BF16 storage differs: " + requirement.name,
      )
    }
    indices[requirement.name] = bindings.length()
    bindings.push({
      name: requirement.name,
      shape: requirement.shape,
      component: requirement.component,
      shard_index: shard,
      tensor_index: index,
      file_offset: 8L +
      shards[shard].encoded_header_bytes().to_int64() +
      tensor.data_start(),
      byte_length: tensor.byte_length(),
    })
    total += tensor.byte_length()
  }
  {
    bindings: ReadOnlyArray::from_array(bindings),
    indices,
    total_bytes: total,
  }
}
