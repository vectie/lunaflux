///|
priv struct Glm53ExpectedHostTensor {
  ordinal : Int
  name : String
  dtype : @glm53_weights.Glm53TensorDType
  shape : ReadOnlyArray[Int64]
}

///|
priv struct Glm53HostTensorLocation {
  shard_ordinal : Int
  tensor_ordinal : Int
}

///|
priv struct Glm53HostTransfer {
  ordinal : Int
  location : Glm53HostTensorLocation
  byte_length : Int64
  spans : ReadOnlyArray[Glm53HostTensorSpan]
}

///|
priv struct Glm53PreparedHostMaterialization {
  transfers : ReadOnlyArray[Glm53HostTransfer]
  regions : ReadOnlyArray[Glm53HostTensorRegion]
  arenas : Array[Glm53HostArena]
  total_bytes : Int64
}

///|
fn validate_expected_digests(
  manifest : @glm53_weights.Glm53TensorManifest,
  numeric : @glm53_numeric_plan.Glm53NumericPlan,
  expected : Glm53HostExpectedDigests,
) -> Unit raise Glm53HostMaterializeError {
  if manifest.encoding() != BFloat16WithFloat32Routing ||
    numeric.encoding() != BFloat16WithFloat32Routing {
    raise UnsupportedPhysicalEncoding
  }
  if manifest.profile() != numeric.profile() {
    raise ProfileMismatch
  }
  if manifest.execution_identity() != expected.execution_identity ||
    numeric.execution_identity() != expected.execution_identity ||
    manifest.execution_identity() != numeric.execution_identity() {
    raise DigestMismatch(ExecutionIdentity)
  }
  if manifest.layout_digest() != expected.manifest_layout ||
    numeric.manifest_layout_digest() != expected.manifest_layout {
    raise DigestMismatch(ManifestLayout)
  }
  if numeric.numeric_schema_digest() != expected.numeric_schema {
    raise DigestMismatch(NumericSchema)
  }
  if numeric.binding_digest() != expected.numeric_binding {
    raise DigestMismatch(NumericBinding)
  }
  if manifest.tensor_count() != numeric.tensor_count() ||
    numeric.numeric_schema().tensors().length() != manifest.tensor_count() {
    raise TensorCountMismatch(
      expected=manifest.tensor_count(),
      actual=numeric.tensor_count(),
    )
  }
}

///|
fn expected_host_tensors(
  manifest : @glm53_weights.Glm53TensorManifest,
  numeric : @glm53_numeric_plan.Glm53NumericPlan,
) -> Array[Glm53ExpectedHostTensor] raise Glm53HostMaterializeError {
  let schema_tensors = numeric.numeric_schema().tensors()
  let expected : Array[Glm53ExpectedHostTensor] = []
  for ordinal in 0..<manifest.tensor_count() {
    guard manifest.tensor(ordinal) is Some(descriptor) &&
      numeric.tensor(ordinal) is Some(binding) &&
      schema_tensors.get(ordinal) is Some(schema) else {
      raise NumericTensorMismatch(ordinal~)
    }
    if descriptor.ordinal() != ordinal ||
      binding.ordinal() != ordinal ||
      schema.ordinal().as_int() != ordinal ||
      schema.shape().dimensions() != descriptor.shape() {
      raise NumericTensorMismatch(ordinal~)
    }
    let storage = schema.storage().dtype()
    match descriptor.dtype() {
      BFloat16 if storage == @numeric_contract.StorageDType::bfloat16() => ()
      Float32 if storage == @numeric_contract.StorageDType::f32() => ()
      Fp8E4M3 => raise UnsupportedPhysicalEncoding
      _ => raise NumericTensorMismatch(ordinal~)
    }
    expected.push({
      ordinal,
      name: descriptor.name(),
      dtype: descriptor.dtype(),
      shape: descriptor.shape(),
    })
  }
  expected
}

///|
fn dtype_byte_width(
  dtype : @glm53_weights.Glm53TensorDType,
) -> Int64 raise Glm53HostMaterializeError {
  match dtype {
    BFloat16 => 2L
    Float32 => 4L
    Fp8E4M3 => raise UnsupportedPhysicalEncoding
  }
}

///|
fn expected_tensor_bytes(
  tensor : Glm53ExpectedHostTensor,
) -> Int64 raise Glm53HostMaterializeError {
  let mut elements = 1L
  for dimension in tensor.shape {
    if dimension <= 0L || elements > 9223372036854775807L / dimension {
      raise TensorByteCountMismatch(ordinal=tensor.ordinal)
    }
    elements *= dimension
  }
  let width = dtype_byte_width(tensor.dtype)
  if elements > 9223372036854775807L / width {
    raise TensorByteCountMismatch(ordinal=tensor.ordinal)
  }
  elements * width
}

///|
fn streaming_dtype_matches(
  expected : @glm53_weights.Glm53TensorDType,
  actual : @streaming_safetensors.StreamingSafetensorsDTypeSpec,
) -> Bool {
  match expected {
    BFloat16 =>
      actual == @streaming_safetensors.StreamingSafetensorsDTypeSpec::bfloat16()
    Float32 =>
      actual == @streaming_safetensors.StreamingSafetensorsDTypeSpec::float32()
    Fp8E4M3 => false
  }
}

///|
fn validate_payload_coverage(
  inspection : @streaming_safetensors.StreamingSafetensorsInspection,
) -> Unit raise Glm53HostMaterializeError {
  for shard_ordinal in 0..<inspection.shard_count() {
    guard inspection.shard(shard_ordinal) is Some(shard) else {
      raise StreamingAdmission(InvalidCopyRequest(shard_ordinal))
    }
    let ranges : Array[(Int64, Int64)] = []
    for tensor_ordinal in 0..<shard.tensor_count() {
      guard shard.tensor(tensor_ordinal) is Some(tensor) else {
        raise StreamingAdmission(InvalidCopyRequest(tensor_ordinal))
      }
      ranges.push((tensor.data_start(), tensor.data_end()))
    }
    ranges.sort_by((left, right) => {
      let (left_start, _) = left
      let (right_start, _) = right
      if left_start < right_start {
        -1
      } else if left_start > right_start {
        1
      } else {
        0
      }
    })
    let mut cursor = 0L
    for range in ranges {
      let (start, end) = range
      if start != cursor {
        raise PayloadCoverageMismatch(shard=shard_ordinal)
      }
      cursor = end
    }
    if cursor != shard.payload_byte_count() {
      raise PayloadCoverageMismatch(shard=shard_ordinal)
    }
  }
}

///|
fn allocate_host_layout(
  expected : ArrayView[Glm53ExpectedHostTensor],
  locations : ArrayView[Glm53HostTensorLocation],
  limits : Glm53HostMaterializationLimits,
) -> Glm53PreparedHostMaterialization raise Glm53HostMaterializeError {
  if expected.length() != locations.length() {
    raise TensorCountMismatch(
      expected=expected.length(),
      actual=locations.length(),
    )
  }
  let byte_lengths : Array[Int64] = []
  let mut total_bytes = 0L
  for tensor in expected {
    let byte_length = expected_tensor_bytes(tensor)
    if total_bytes > limits.max_total_arena_bytes - byte_length {
      raise ArenaSizeOverflow
    }
    total_bytes += byte_length
    byte_lengths.push(byte_length)
  }
  let arenas : Array[Glm53HostArena] = []
  let mut remaining_total = total_bytes
  while remaining_total > 0L {
    if arenas.length() >= 65536 {
      raise ArenaSizeOverflow
    }
    let size = if remaining_total < limits.max_arena_bytes.to_int64() {
      remaining_total.to_int()
    } else {
      limits.max_arena_bytes
    }
    arenas.push({ bytes: FixedArray::make(size, b'\x00'), })
    remaining_total -= size.to_int64()
  }
  let transfers : Array[Glm53HostTransfer] = []
  let regions : Array[Glm53HostTensorRegion] = []
  let mut arena_index = 0
  let mut arena_offset = 0
  for ordinal, tensor in expected {
    let byte_length = byte_lengths[ordinal]
    let spans : Array[Glm53HostTensorSpan] = []
    let mut remaining = byte_length
    while remaining > 0L {
      let available = arenas[arena_index].bytes.length() - arena_offset
      let count = if remaining < available.to_int64() {
        remaining.to_int()
      } else {
        available
      }
      spans.push({
        arena_index,
        offset_bytes: arena_offset,
        byte_length: count,
      })
      remaining -= count.to_int64()
      arena_offset += count
      if arena_offset == arenas[arena_index].bytes.length() {
        arena_index += 1
        arena_offset = 0
      }
    }
    let readonly_spans = ReadOnlyArray::from_array(spans)
    transfers.push({
      ordinal: tensor.ordinal,
      location: locations[ordinal],
      byte_length,
      spans: readonly_spans,
    })
    regions.push({
      ordinal: tensor.ordinal,
      byte_length,
      spans: readonly_spans,
    })
  }
  {
    transfers: ReadOnlyArray::from_array(transfers),
    regions: ReadOnlyArray::from_array(regions),
    arenas,
    total_bytes,
  }
}

///|
fn prepare_expected_host_materialization(
  expected : ArrayView[Glm53ExpectedHostTensor],
  inspection : @streaming_safetensors.StreamingSafetensorsInspection,
  limits : Glm53HostMaterializationLimits,
) -> Glm53PreparedHostMaterialization raise Glm53HostMaterializeError {
  if inspection.total_tensor_count() != expected.length() {
    raise TensorCountMismatch(
      expected=expected.length(),
      actual=inspection.total_tensor_count(),
    )
  }
  validate_payload_coverage(inspection)
  let locations_by_name : Map[String, Glm53HostTensorLocation] = Map(
    [],
    capacity=expected.length(),
  )
  for shard_ordinal in 0..<inspection.shard_count() {
    guard inspection.shard(shard_ordinal) is Some(shard) else {
      raise StreamingAdmission(InvalidCopyRequest(shard_ordinal))
    }
    for tensor_ordinal in 0..<shard.tensor_count() {
      guard shard.tensor(tensor_ordinal) is Some(tensor) else {
        raise StreamingAdmission(InvalidCopyRequest(tensor_ordinal))
      }
      locations_by_name.set(tensor.name(), { shard_ordinal, tensor_ordinal, })
    }
  }
  let locations : Array[Glm53HostTensorLocation] = []
  for expected_tensor in expected {
    guard locations_by_name.get(expected_tensor.name) is Some(location) else {
      raise TensorNameMismatch(ordinal=expected_tensor.ordinal)
    }
    guard inspection.shard(location.shard_ordinal) is Some(shard) &&
      shard.tensor(location.tensor_ordinal) is Some(actual) else {
      raise TensorNameMismatch(ordinal=expected_tensor.ordinal)
    }
    if !streaming_dtype_matches(expected_tensor.dtype, actual.dtype()) {
      raise TensorDTypeMismatch(ordinal=expected_tensor.ordinal)
    }
    if actual.shape() != expected_tensor.shape {
      raise TensorShapeMismatch(ordinal=expected_tensor.ordinal)
    }
    if actual.byte_count() != expected_tensor_bytes(expected_tensor) {
      raise TensorByteCountMismatch(ordinal=expected_tensor.ordinal)
    }
    locations.push(location)
  }
  allocate_host_layout(expected, locations, limits)
}

///|
fn prepare_host_materialization(
  manifest : @glm53_weights.Glm53TensorManifest,
  numeric : @glm53_numeric_plan.Glm53NumericPlan,
  expected : Glm53HostExpectedDigests,
  inspection : @streaming_safetensors.StreamingSafetensorsInspection,
  limits : Glm53HostMaterializationLimits,
) -> Glm53PreparedHostMaterialization raise Glm53HostMaterializeError {
  validate_expected_digests(manifest, numeric, expected)
  prepare_expected_host_materialization(
    expected_host_tensors(manifest, numeric),
    inspection,
    limits,
  )
}
