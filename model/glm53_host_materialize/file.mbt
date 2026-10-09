///|
fn zero_arenas(arenas : ArrayView[Glm53HostArena]) -> Unit {
  for arena in arenas {
    for index in 0..<arena.bytes.length() {
      arena.bytes[index] = b'\x00'
    }
  }
}

///|
fn streaming_limits(
  limits : Glm53HostMaterializationLimits,
) -> @streaming_safetensors.StreamingSafetensorsLimits raise Glm53HostMaterializeError {
  let maximum_copy_requests = if limits.max_tensor_count > 99999 {
    165535
  } else {
    limits.max_tensor_count + 65536
  }
  @streaming_safetensors.StreamingSafetensorsLimits::new(
    maximum_shards=limits.max_source_count,
    maximum_file_bytes=limits.max_file_bytes,
    maximum_header_bytes=limits.max_header_bytes,
    maximum_tensors_per_shard=limits.max_tensor_count,
    maximum_total_tensors=limits.max_tensor_count,
    maximum_rank=limits.max_rank,
    maximum_name_bytes=limits.max_name_bytes,
    maximum_dtype_specs=3,
    authentication_chunk_bytes=1024 * 1024,
    maximum_copy_requests~,
    maximum_copy_bytes=limits.max_total_arena_bytes,
  ) catch {
    error => raise StreamingAdmission(error)
  }
}

///|
fn streaming_dtype_specs() -> Array[
  @streaming_safetensors.StreamingSafetensorsDTypeSpec,
] raise Glm53HostMaterializeError {
  [
    @streaming_safetensors.StreamingSafetensorsDTypeSpec::bfloat16(),
    @streaming_safetensors.StreamingSafetensorsDTypeSpec::float32(),
    @streaming_safetensors.StreamingSafetensorsDTypeSpec::opaque_one_byte(
      "F8_E4M3",
    ) catch {
      error => raise StreamingAdmission(error)
    },
  ]
}

///|
fn streaming_inputs(
  sources : ArrayView[Glm53HostShardSource],
) -> Array[@streaming_safetensors.StreamingSafetensorsShardInput] {
  let inputs : Array[@streaming_safetensors.StreamingSafetensorsShardInput] = []
  for source in sources {
    inputs.push(
      @streaming_safetensors.StreamingSafetensorsShardInput::new(
        locator=source.locator,
        expected_digest=source.artifact_digest,
      ),
    )
  }
  inputs
}

///|
fn inspect_host_shards(
  root : @approved_fs.ApprovedRoot,
  sources : ArrayView[Glm53HostShardSource],
  limits : Glm53HostMaterializationLimits,
) -> @streaming_safetensors.StreamingSafetensorsInspection raise Glm53HostMaterializeError {
  let neutral_limits = streaming_limits(limits)
  let inputs = streaming_inputs(sources)
  let dtype_specs = streaming_dtype_specs()
  @streaming_safetensors.inspect_shards(
    root,
    inputs,
    dtype_specs~,
    limits=neutral_limits,
  ) catch {
    error => raise StreamingAdmission(error)
  }
}

///|
fn segmented_destinations(
  arenas : ArrayView[Glm53HostArena],
) -> Array[FixedArray[Byte]] {
  let destinations : Array[FixedArray[Byte]] = []
  for arena in arenas {
    destinations.push(arena.bytes)
  }
  destinations
}

///|
fn segmented_requests(
  transfers : ArrayView[Glm53HostTransfer],
) -> Array[@streaming_safetensors.StreamingSafetensorsSegmentedCopyRequest] raise Glm53HostMaterializeError {
  let requests : Array[
    @streaming_safetensors.StreamingSafetensorsSegmentedCopyRequest,
  ] = []
  for transfer in transfers {
    let mut tensor_byte_offset = 0L
    for span in transfer.spans {
      requests.push(
        @streaming_safetensors.StreamingSafetensorsSegmentedCopyRequest::new(
          shard_ordinal=transfer.location.shard_ordinal,
          tensor_ordinal=transfer.location.tensor_ordinal,
          tensor_byte_offset~,
          destination_arena_ordinal=span.arena_index,
          destination_offset=span.offset_bytes,
          byte_count=span.byte_length,
        ) catch {
          error => raise StreamingAdmission(error)
        },
      )
      tensor_byte_offset += span.byte_length.to_int64()
    }
    if tensor_byte_offset != transfer.byte_length {
      raise TensorByteCountMismatch(ordinal=transfer.ordinal)
    }
  }
  requests
}

///|
fn visit_matches_transfer(
  transfers : ArrayView[Glm53HostTransfer],
  visited : @streaming_safetensors.StreamingSafetensorsSegmentedVisitedRange,
  seen : Array[Int],
) -> Unit raise @streaming_safetensors.StreamingSafetensorsError {
  let mut request_ordinal = 0
  for transfer in transfers {
    let mut tensor_byte_offset = 0L
    for span in transfer.spans {
      if transfer.location.shard_ordinal == visited.shard_ordinal() &&
        transfer.location.tensor_ordinal == visited.tensor_ordinal() &&
        tensor_byte_offset == visited.tensor_byte_offset() &&
        span.arena_index == visited.destination_arena_ordinal() &&
        span.offset_bytes == visited.destination_offset() &&
        span.byte_length == visited.byte_count() {
        if seen[request_ordinal] != 0 {
          raise VisitorRejected(request_ordinal)
        }
        seen[request_ordinal] = 1
        return
      }
      tensor_byte_offset += span.byte_length.to_int64()
      request_ordinal += 1
    }
  }
  raise VisitorRejected(request_ordinal)
}

///|
fn run_segmented_copy_with_cleanup(
  arenas : ArrayView[Glm53HostArena],
  copy : () -> Unit raise @streaming_safetensors.StreamingSafetensorsError,
) -> Unit raise Glm53HostMaterializeError {
  copy() catch {
    error => {
      zero_arenas(arenas)
      raise StreamingAdmission(error)
    }
  }
}

///|
fn copy_prepared_materialization(
  root : @approved_fs.ApprovedRoot,
  inspection : @streaming_safetensors.StreamingSafetensorsInspection,
  prepared : Glm53PreparedHostMaterialization,
  limits : Glm53HostMaterializationLimits,
) -> Unit raise Glm53HostMaterializeError {
  let destinations = segmented_destinations(prepared.arenas)
  let requests = segmented_requests(prepared.transfers)
  let seen = Array::make(requests.length(), 0)
  let neutral_limits = streaming_limits(limits)
  run_segmented_copy_with_cleanup(prepared.arenas, () => {
    @streaming_safetensors.copy_segmented_ranges(
      root,
      inspection,
      destination_arenas=destinations,
      requests~,
      limits=neutral_limits,
      visit=(_, visited) => {
        visit_matches_transfer(prepared.transfers, visited, seen)
      },
    )
  })
  for request_ordinal, count in seen {
    if count != 1 {
      zero_arenas(prepared.arenas)
      raise StreamingAdmission(VisitorRejected(request_ordinal))
    }
  }
}

///|
/// Startup-only exact mixed BF16/F32 materialization from authenticated
/// safetensors shards. The neutral reader inspects and authenticates each
/// shard, then one segmented operation copies all tensor spans directly into
/// their bounded final arenas.
pub fn materialize_safetensors_shards(
  root : @approved_fs.ApprovedRoot,
  sources : ArrayView[Glm53HostShardSource],
  manifest : @glm53_weights.Glm53TensorManifest,
  numeric : @glm53_numeric_plan.Glm53NumericPlan,
  expected~ : Glm53HostExpectedDigests,
  limits~ : Glm53HostMaterializationLimits,
) -> Glm53HostWeights raise Glm53HostMaterializeError {
  if sources.is_empty() || sources.length() > limits.max_source_count {
    raise SourceCountMismatch
  }
  validate_expected_digests(manifest, numeric, expected)
  let inspection = inspect_host_shards(root, sources, limits)
  if inspection.shard_count() != sources.length() {
    raise SourceCountMismatch
  }
  let prepared = prepare_host_materialization(
    manifest, numeric, expected, inspection, limits,
  )
  copy_prepared_materialization(root, inspection, prepared, limits)
  let source_digest = canonical_host_source_digest(expected, sources, prepared)
  {
    execution_identity: expected.execution_identity,
    manifest_layout: expected.manifest_layout,
    numeric_schema: expected.numeric_schema,
    numeric_binding: expected.numeric_binding,
    source_digest,
    regions: prepared.regions,
    arenas: prepared.arenas,
    total_bytes: prepared.total_bytes,
    open: true,
  }
}
