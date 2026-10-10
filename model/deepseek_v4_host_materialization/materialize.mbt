///|
fn drop_arenas(arenas : Array[FixedArray[Byte]?]) -> Unit {
  for arena_index in 0..<arenas.length() {
    arenas[arena_index] = None
  }
}

///|
fn validate_manifest_replay(
  manifest : DeepSeekV4HostManifest,
  expected_digests : DeepSeekV4HostExpectedDigests,
  expected_manifest_digest : DeepSeekV4HostManifestDigest,
) -> Unit raise DeepSeekV4HostMaterializationError {
  if manifest.identity != expected_digests.identity ||
    manifest.binding.identity() != expected_digests.identity ||
    manifest.numeric_plan.identity() != expected_digests.identity {
    raise IdentityMismatch
  }
  if manifest.expected_digests != expected_digests {
    raise IdentityMismatch
  }
  if manifest.numeric_plan.logical_manifest_digest() !=
    expected_digests.logical_manifest_digest {
    raise LogicalManifestDigestMismatch
  }
  if manifest.numeric_plan.numeric_schema_digest() !=
    expected_digests.numeric_schema_digest {
    raise NumericSchemaDigestMismatch
  }
  if manifest.numeric_plan.binding_digest() !=
    expected_digests.numeric_binding_digest {
    raise NumericBindingDigestMismatch
  }
  // build_manifest already joined the immutable binding and numeric plan.
  // Loading consumes that plan and label; it must not repeat the complete
  // numeric scan or canonical manifest serialization/hash.
  if manifest.digest != expected_manifest_digest {
    raise ManifestDigestMismatch
  }
}

///|
fn allocate_final_arenas(
  sizes : ReadOnlyArray[Int],
) -> (Array[FixedArray[Byte]?], Array[FixedArray[Byte]]) {
  let owners : Array[FixedArray[Byte]?] = []
  let destinations : Array[FixedArray[Byte]] = []
  for size in sizes {
    let arena = FixedArray::make(size, b'\x00')
    owners.push(Some(arena))
    destinations.push(arena)
  }
  (owners, destinations)
}

///|
fn streaming_copy_requests(
  tensors : ReadOnlyArray[DeepSeekV4HostLayoutTensor],
) -> Array[@streaming_safetensors.StreamingSafetensorsSegmentedCopyRequest] raise DeepSeekV4HostMaterializationError {
  let requests : Array[
    @streaming_safetensors.StreamingSafetensorsSegmentedCopyRequest,
  ] = []
  for tensor in tensors {
    requests.push(
      @streaming_safetensors.StreamingSafetensorsSegmentedCopyRequest::new(
        shard_ordinal=tensor.shard_ordinal,
        tensor_ordinal=tensor.tensor_ordinal,
        tensor_byte_offset=0L,
        destination_arena_ordinal=tensor.arena_ordinal,
        destination_offset=tensor.arena_offset,
        byte_count=tensor.byte_count,
      ) catch {
        _ => raise StreamingFailure
      },
    )
  }
  requests
}

///|
fn copy_layout_into_final_arenas(
  root : @approved_fs.ApprovedRoot,
  inspection : @streaming_safetensors.StreamingSafetensorsInspection,
  tensors : ReadOnlyArray[DeepSeekV4HostLayoutTensor],
  arena_sizes : ReadOnlyArray[Int],
  streaming_limits : @streaming_safetensors.StreamingSafetensorsLimits,
) -> Array[FixedArray[Byte]?] raise DeepSeekV4HostMaterializationError {
  let (owners, destinations) = allocate_final_arenas(arena_sizes)
  let requests = streaming_copy_requests(tensors)
  let mut visited = 0
  @streaming_safetensors.copy_segmented_ranges(
    root,
    inspection,
    destination_arenas=destinations,
    requests~,
    limits=streaming_limits,
    visit=(_, _) => visited += 1,
  ) catch {
    _ => {
      drop_arenas(owners)
      raise StreamingFailure
    }
  }
  if visited != tensors.length() {
    drop_arenas(owners)
    raise StreamingFailure
  }
  owners
}

///|
/// Copy selected raw tensors into final bounded arenas using the retained
/// immutable plan. Header/range checks remain; no payload or manifest rehash
/// occurs here, and no dtype interpretation is performed.
pub fn materialize(
  root : @approved_fs.ApprovedRoot,
  manifest : DeepSeekV4HostManifest,
  expected_digests~ : DeepSeekV4HostExpectedDigests,
  expected_manifest_digest~ : DeepSeekV4HostManifestDigest,
) -> DeepSeekV4HostWeights raise DeepSeekV4HostMaterializationError {
  validate_manifest_replay(manifest, expected_digests, expected_manifest_digest)
  let owners = copy_layout_into_final_arenas(
    root,
    manifest.inspection,
    manifest.tensors,
    manifest.arena_sizes,
    manifest.streaming_limits,
  )
  let tensors : Array[DeepSeekV4HostTensor] = []
  for layout in manifest.tensors {
    guard manifest.binding.tensor(layout.binding_index) is Some(binding) else {
      drop_arenas(owners)
      raise WeightBindingFailure
    }
    tensors.push({
      binding,
      arena_ordinal: layout.arena_ordinal,
      arena_offset: layout.arena_offset,
      byte_count: layout.byte_count,
      quantized_parameter_index: layout.quantized_parameter_index,
    })
  }
  {
    identity: manifest.identity,
    expected_digests: manifest.expected_digests,
    manifest_digest: manifest.digest,
    tensors: ReadOnlyArray::from_array(tensors),
    arenas: owners,
    total_tensor_bytes: manifest.total_tensor_bytes,
    total_arena_bytes: manifest.total_arena_bytes,
    released: false,
  }
}

///|
/// Drop all arena references and invalidate byte access without scrubbing bytes.
/// Storage remains GC-managed; no operating-system release timing is claimed.
pub fn DeepSeekV4HostWeights::release(self : Self) -> Bool {
  if self.released {
    return false
  }
  drop_arenas(self.arenas)
  self.released = true
  true
}
