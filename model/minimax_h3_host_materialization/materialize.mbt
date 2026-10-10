///|
fn drop_arenas(arenas : Array[FixedArray[Byte]?]) -> Unit {
  for index in 0..<arenas.length() {
    arenas[index] = None
  }
}

///|
fn allocate_arenas(
  manifest : MiniMaxH3HostComponentManifest,
) -> Array[FixedArray[Byte]?] {
  let arenas : Array[FixedArray[Byte]?] = []
  for size in manifest.arena_sizes {
    arenas.push(Some(FixedArray::make(size.to_int(), b'\x00')))
  }
  arenas
}

///|
fn validate_materialization_admission(
  manifest : MiniMaxH3HostComponentManifest,
  expected_identity : @spec.ModelIdentity,
  expected_manifest_digest : MiniMaxH3HostManifestDigest,
  limits : MiniMaxH3HostMaterializationLimits,
) -> Unit raise MiniMaxH3HostMaterializationError {
  require_materializable_component(manifest.component)
  if manifest.identity != expected_identity {
    raise IdentityMismatch
  }
  let recomputed = canonical_manifest_digest(
    manifest.component,
    manifest.identity,
    manifest.shards,
    manifest.tensors,
    manifest.arena_sizes,
    manifest.total_tensor_bytes,
  )
  if manifest.digest != expected_manifest_digest ||
    recomputed != manifest.digest {
    raise ManifestDigestMismatch
  }
  if manifest.shards.is_empty() ||
    manifest.shards.length() > limits.maximum_shards ||
    manifest.tensors.length() > limits.maximum_tensors {
    raise InvalidShardCount
  }
  if manifest.arena_sizes.length() > limits.maximum_arenas {
    raise ArenaCountExceeded
  }
  for size in manifest.arena_sizes {
    if size <= 0L || size > limits.maximum_arena_bytes {
      raise HostBudgetExceeded
    }
  }
  ignore(total_sizes(manifest.arena_sizes, limits.maximum_total_host_bytes))
  for shard_index, shard in manifest.shards {
    if shard.payload_bytes > limits.maximum_payload_bytes_per_shard {
      raise ShardPayloadMismatch(shard_index)
    }
  }
}

///|
fn streaming_destinations(
  arenas : Array[FixedArray[Byte]?],
) -> Array[FixedArray[Byte]] raise MiniMaxH3HostMaterializationError {
  let destinations : Array[FixedArray[Byte]] = []
  for arena in arenas {
    guard arena is Some(bytes) else { raise ArenaReleased }
    destinations.push(bytes)
  }
  destinations
}

///|
fn streaming_copy_requests(
  manifest : MiniMaxH3HostComponentManifest,
  locations : ReadOnlyArray[MiniMaxH3StreamingTensorLocation],
) -> Array[@streaming_safetensors.StreamingSafetensorsSegmentedCopyRequest] raise MiniMaxH3HostMaterializationError {
  if locations.length() != manifest.tensors.length() {
    raise StreamingTensorCountMismatch(-1)
  }
  let requests : Array[
    @streaming_safetensors.StreamingSafetensorsSegmentedCopyRequest,
  ] = []
  for binding_index, entry in manifest.tensors {
    let location = locations[binding_index]
    let request = @streaming_safetensors.StreamingSafetensorsSegmentedCopyRequest::new(
      shard_ordinal=location.shard_ordinal,
      tensor_ordinal=location.tensor_ordinal,
      tensor_byte_offset=0L,
      destination_arena_ordinal=entry.arena_index,
      destination_offset=entry.arena_offset.to_int(),
      byte_count=entry.tensor.byte_length().to_int(),
    ) catch {
      _ => raise StreamingFailure
    }
    requests.push(request)
  }
  requests
}

///|
fn visit_matches_manifest(
  manifest : MiniMaxH3HostComponentManifest,
  locations : ReadOnlyArray[MiniMaxH3StreamingTensorLocation],
  tensor : @streaming_safetensors.StreamingSafetensorsTensorMetadata,
  visited : @streaming_safetensors.StreamingSafetensorsSegmentedVisitedRange,
  seen : Array[Bool],
) -> Unit raise @streaming_safetensors.StreamingSafetensorsError {
  let mut binding_index = -1
  for index, location in locations {
    if location.shard_ordinal == visited.shard_ordinal() &&
      location.tensor_ordinal == visited.tensor_ordinal() {
      if binding_index >= 0 {
        raise VisitorRejected(index)
      }
      binding_index = index
    }
  }
  if binding_index < 0 || seen[binding_index] {
    raise VisitorRejected(binding_index)
  }
  let expected = manifest.tensors[binding_index]
  if tensor.name() != expected.tensor.name() ||
    tensor.dtype() != streaming_dtype(expected.tensor.dtype()) ||
    tensor.shape() != expected.tensor.shape() ||
    tensor.data_start() != expected.tensor.data_start() ||
    tensor.data_end() != expected.tensor.data_end() ||
    visited.tensor_byte_offset() != 0L ||
    visited.destination_arena_ordinal() != expected.arena_index ||
    visited.destination_offset() != expected.arena_offset.to_int() ||
    visited.byte_count() != expected.tensor.byte_length().to_int() {
    raise VisitorRejected(binding_index)
  }
  seen[binding_index] = true
}

///|
/// Startup-only exact mixed-precision materialization from authenticated streaming
/// safetensors shards. Each shard is reauthenticated once while tensor bytes
/// are copied directly into final bounded arenas.
pub fn materialize_safetensors_shards(
  root : @approved_fs.ApprovedRoot,
  sources : ArrayView[MiniMaxH3ShardSource],
  manifest : MiniMaxH3HostComponentManifest,
  expected_identity~ : @spec.ModelIdentity,
  expected_manifest_digest~ : MiniMaxH3HostManifestDigest,
  limits~ : MiniMaxH3HostMaterializationLimits,
) -> MiniMaxH3HostComponentArena raise MiniMaxH3HostMaterializationError {
  validate_materialization_admission(
    manifest, expected_identity, expected_manifest_digest, limits,
  )
  let admission = inspect_component_shards(root, sources, manifest, limits)
  let arenas = allocate_arenas(manifest)
  let destinations = streaming_destinations(arenas)
  let requests = streaming_copy_requests(manifest, admission.locations)
  let seen = Array::make(manifest.tensors.length(), false)
  let copied : Result[Unit, Error] = Ok(
    @streaming_safetensors.copy_segmented_ranges(
      root,
      admission.inspection,
      destination_arenas=destinations,
      requests~,
      limits=streaming_limits(limits),
      visit=(tensor, range) => {
        visit_matches_manifest(
          manifest,
          admission.locations,
          tensor,
          range,
          seen,
        )
      },
    ),
  ) catch {
    error => Err(error)
  }
  match copied {
    Err(_) => {
      drop_arenas(arenas)
      raise StreamingFailure
    }
    Ok(_) => {
      for binding_index, was_seen in seen {
        if !was_seen {
          drop_arenas(arenas)
          raise StreamingTensorMismatch(binding_index)
        }
      }
      let tensors : Array[MiniMaxH3HostTensor] = []
      for entry in manifest.tensors {
        tensors.push({
          name: entry.tensor.name(),
          role: entry.role,
          shape: entry.tensor.shape(),
          arena_index: entry.arena_index,
          arena_offset: entry.arena_offset,
          byte_length: entry.tensor.byte_length(),
        })
      }
      {
        component: manifest.component,
        identity: manifest.identity,
        manifest_digest: manifest.digest,
        tensors: ReadOnlyArray::from_array(tensors),
        arenas,
        total_tensor_bytes: manifest.total_tensor_bytes,
        released: false,
      }
    }
  }
}
