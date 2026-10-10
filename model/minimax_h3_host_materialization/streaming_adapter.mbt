///|
priv struct MiniMaxH3StreamingTensorLocation {
  shard_ordinal : Int
  tensor_ordinal : Int
}

///|
priv struct MiniMaxH3StreamingAdmission {
  inspection : @streaming_safetensors.StreamingSafetensorsInspection
  locations : ReadOnlyArray[MiniMaxH3StreamingTensorLocation]
}

///|
fn streaming_limits(
  limits : MiniMaxH3HostMaterializationLimits,
) -> @streaming_safetensors.StreamingSafetensorsLimits raise MiniMaxH3HostMaterializationError {
  @streaming_safetensors.StreamingSafetensorsLimits::new(
    maximum_shards=limits.maximum_shards,
    maximum_file_bytes=limits.maximum_payload_bytes_per_shard +
      limits.maximum_header_bytes.to_int64() +
      8L,
    maximum_header_bytes=limits.maximum_header_bytes,
    maximum_tensors_per_shard=limits.maximum_tensors,
    maximum_total_tensors=limits.maximum_tensors,
    maximum_rank=16,
    maximum_name_bytes=limits.maximum_name_bytes,
    maximum_dtype_specs=2,
    authentication_chunk_bytes=1048576,
    // Model source layouts can split one canonical projection into 32 head
    // slices. This bounds descriptor count, not payload or resident memory.
    maximum_copy_requests=limits.maximum_tensors * 32,
    maximum_copy_bytes=limits.maximum_total_host_bytes,
  ) catch {
    _ => raise StreamingFailure
  }
}

///|
fn streaming_inputs(
  sources : ArrayView[MiniMaxH3ShardSource],
  manifest : MiniMaxH3HostComponentManifest,
) -> Array[@streaming_safetensors.StreamingSafetensorsShardInput] raise MiniMaxH3HostMaterializationError {
  if sources.length() != manifest.shards.length() {
    raise ShardCountMismatch
  }
  let inputs : Array[@streaming_safetensors.StreamingSafetensorsShardInput] = []
  for shard_ordinal, source in sources {
    let digest = @streaming_safetensors.StreamingSafetensorsFileDigest::from_sha256(
      manifest.shards[shard_ordinal].file_digest.as_hex(),
    ) catch {
      _ => raise StreamingFailure
    }
    inputs.push(
      @streaming_safetensors.StreamingSafetensorsShardInput::new(
        locator=source.locator,
        expected_digest=digest,
      ),
    )
  }
  inputs
}

///|
fn find_streaming_tensor(
  shard : @streaming_safetensors.StreamingSafetensorsShardMetadata,
  name : String,
) -> (Int, @streaming_safetensors.StreamingSafetensorsTensorMetadata)? {
  for tensor_ordinal in 0..<shard.tensor_count() {
    guard shard.tensor(tensor_ordinal) is Some(tensor) else { continue }
    if tensor.name() == name {
      return Some((tensor_ordinal, tensor))
    }
  }
  None
}

///|
fn streaming_dtype(
  dtype : @safetensors.TensorDType,
) -> @streaming_safetensors.StreamingSafetensorsDTypeSpec {
  match dtype {
    BFloat16 => @streaming_safetensors.StreamingSafetensorsDTypeSpec::bfloat16()
    Float32 => @streaming_safetensors.StreamingSafetensorsDTypeSpec::float32()
  }
}

///|
fn validate_streaming_inspection(
  inspection : @streaming_safetensors.StreamingSafetensorsInspection,
  manifest : MiniMaxH3HostComponentManifest,
) -> ReadOnlyArray[MiniMaxH3StreamingTensorLocation] raise MiniMaxH3HostMaterializationError {
  if inspection.shard_count() != manifest.shards.length() {
    raise ShardCountMismatch
  }
  if inspection.total_tensor_count() != manifest.tensors.length() {
    raise StreamingTensorCountMismatch(-1)
  }
  for shard_ordinal, declaration in manifest.shards {
    guard inspection.shard(shard_ordinal) is Some(shard) else {
      raise StreamingShardMismatch(shard_ordinal)
    }
    if shard.file_digest().as_hex() != declaration.file_digest.as_hex() {
      raise StreamingShardMismatch(shard_ordinal)
    }
    if shard.payload_byte_count() != declaration.payload_bytes {
      raise ShardPayloadMismatch(shard_ordinal)
    }
    let mut expected_count = 0
    for entry in manifest.tensors {
      if entry.shard_index == shard_ordinal {
        expected_count += 1
      }
    }
    if shard.tensor_count() != expected_count {
      raise StreamingTensorCountMismatch(shard_ordinal)
    }
  }
  let locations : Array[MiniMaxH3StreamingTensorLocation] = []
  for entry in manifest.tensors {
    guard inspection.shard(entry.shard_index) is Some(shard) else {
      raise StreamingShardMismatch(entry.shard_index)
    }
    guard find_streaming_tensor(shard, entry.tensor.name())
      is Some((tensor_ordinal, actual)) else {
      raise StreamingTensorMismatch(entry.binding_index)
    }
    if actual.dtype() != streaming_dtype(entry.tensor.dtype()) ||
      actual.shape() != entry.tensor.shape() ||
      actual.data_start() != entry.tensor.data_start() ||
      actual.data_end() != entry.tensor.data_end() {
      raise StreamingTensorMismatch(entry.binding_index)
    }
    locations.push({ shard_ordinal: entry.shard_index, tensor_ordinal, })
  }
  ReadOnlyArray::from_array(locations)
}

///|
fn inspect_component_shards(
  root : @approved_fs.ApprovedRoot,
  sources : ArrayView[MiniMaxH3ShardSource],
  manifest : MiniMaxH3HostComponentManifest,
  limits : MiniMaxH3HostMaterializationLimits,
) -> MiniMaxH3StreamingAdmission raise MiniMaxH3HostMaterializationError {
  let inputs = streaming_inputs(sources, manifest)
  let inspection = @streaming_safetensors.inspect_shards(
    root,
    inputs,
    dtype_specs=[
      @streaming_safetensors.StreamingSafetensorsDTypeSpec::bfloat16(),
      @streaming_safetensors.StreamingSafetensorsDTypeSpec::float32(),
    ],
    limits=streaming_limits(limits),
  ) catch {
    _ => raise StreamingFailure
  }
  {
    inspection,
    locations: validate_streaming_inspection(inspection, manifest),
  }
}
