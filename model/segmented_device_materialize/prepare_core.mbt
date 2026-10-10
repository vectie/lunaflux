///|
fn validate_layout_replay(
  layout : SegmentedDeviceLayout,
  source_arenas : ArrayView[FixedArray[Byte]],
  expected_identity : @spec.ModelIdentity,
  expected_source_digest : SegmentedHostSourceDigest,
  expected_layout_digest : SegmentedDeviceLayoutDigest,
) -> Unit raise SegmentedDeviceMaterializationError {
  if layout.source_identity != expected_identity {
    raise IdentityMismatch
  }
  if layout.source_digest != expected_source_digest {
    raise SourceDigestMismatch
  }
  if layout.digest != expected_layout_digest {
    raise LayoutDigestMismatch
  }
  if source_arenas.length() != layout.source_arena_lengths.length() {
    raise SourceArenaShapeMismatch(-1)
  }
  for index, arena in source_arenas {
    if arena.length() != layout.source_arena_lengths[index] {
      raise SourceArenaShapeMismatch(index)
    }
  }
}

///|
fn[T] failed_resource(
  resource : T,
  cause : SegmentedDeviceMaterializationError,
  stage : SegmentedDeviceFailureStage,
  close : (T) -> Unit raise @device.DeviceError,
) -> SegmentedResourcePreparation[T] raise SegmentedDeviceMaterializationError {
  try close(resource) catch {
    cleanup_failure =>
      SegmentedResourceCleanupRequired({
        resource,
        cause,
        stage,
        cleanup_failure,
      })
  } noraise {
    _ => raise cause
  }
}

///|
fn[T] prepare_with_resource(
  layout : SegmentedDeviceLayout,
  source_arenas : ArrayView[FixedArray[Byte]],
  expected_identity : @spec.ModelIdentity,
  expected_source_digest : SegmentedHostSourceDigest,
  expected_layout_digest : SegmentedDeviceLayoutDigest,
  allocate : (Int64) -> T raise SegmentedDeviceMaterializationError,
  validate : (T, Int64, Int64, Int64) -> Unit raise SegmentedDeviceMaterializationError,
  copy : (T, FixedArray[Byte], Int64, Int64, Int64) -> Unit raise SegmentedDeviceMaterializationError,
  close : (T) -> Unit raise @device.DeviceError,
) -> SegmentedResourcePreparation[T] raise SegmentedDeviceMaterializationError {
  validate_layout_replay(
    layout, source_arenas, expected_identity, expected_source_digest, expected_layout_digest,
  )
  let resource = allocate(layout.device_arena_bytes)
  for region in layout.regions {
    let outcome : Result[Unit, SegmentedDeviceMaterializationError] = Ok(
      validate(
        resource,
        region.device_offset,
        region.byte_count,
        layout.alignment_bytes,
      ),
    ) catch {
      error => Err(error)
    }
    match outcome {
      Ok(_) => ()
      Err(cause) =>
        return failed_resource(
          resource,
          cause,
          AllocationRegionValidation,
          close,
        )
    }
  }
  for region in layout.regions {
    let source = source_arenas[region.source_arena_ordinal]
    let outcome : Result[Unit, SegmentedDeviceMaterializationError] = Ok(
      copy(
        resource,
        source,
        region.source_offset.to_int64(),
        region.device_offset,
        region.byte_count,
      ),
    ) catch {
      error => Err(error)
    }
    match outcome {
      Ok(_) => ()
      Err(cause) => return failed_resource(resource, cause, Transfer, close)
    }
  }
  SegmentedResourceReady({ resource, layout, })
}
