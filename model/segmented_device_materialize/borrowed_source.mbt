///|
fn translate_borrow_failure(
  error : @segmented_host_borrow.SegmentedHostBorrowError,
) -> SegmentedDeviceMaterializationError {
  match error {
    InvalidConsumer => HostArenaNotProvided(-1)
    BorrowClosed(ordinal) => HostArenaBorrowClosed(ordinal)
    ArenaAlreadyProvided(ordinal) => HostArenaAlreadyProvided(ordinal)
    ArenaLengthMismatch(ordinal) => HostArenaLengthMismatch(ordinal)
    ArenaNotProvided(ordinal) => HostArenaNotProvided(ordinal)
    SourceUnavailable(ordinal) => HostSourceUnavailable(ordinal)
    ConsumerRejected(ordinal) => HostArenaConsumerRejected(ordinal)
  }
}

///|
fn provide_borrowed_region(
  layout : SegmentedDeviceLayout,
  region : SegmentedDeviceTensorRegion,
  provide : (Int, @segmented_host_borrow.SegmentedHostArenaConsumer) -> Unit raise @segmented_host_borrow.SegmentedHostBorrowError,
  consume : (FixedArray[Byte]) -> Unit raise SegmentedDeviceMaterializationError,
) -> Unit raise SegmentedDeviceMaterializationError {
  let arena_ordinal = region.source_arena_ordinal
  let expected_length = layout.source_arena_lengths[arena_ordinal]
  let mut consume_failure : SegmentedDeviceMaterializationError? = None
  let consumer = @segmented_host_borrow.SegmentedHostArenaConsumer::new(
    arena_ordinal~,
    expected_length~,
    consume=source => {
      let outcome : Result[Unit, SegmentedDeviceMaterializationError] = Ok(
        consume(source),
      ) catch {
        error => Err(error)
      }
      match outcome {
        Ok(_) => true
        Err(error) => {
          consume_failure = Some(error)
          false
        }
      }
    },
  ) catch {
    error => raise translate_borrow_failure(error)
  }
  let outcome : Result[Unit, @segmented_host_borrow.SegmentedHostBorrowError] = Ok(
    provide(arena_ordinal, consumer),
  ) catch {
    error => Err(error)
  }
  consumer.revoke()
  match consume_failure {
    Some(error) => raise error
    None => ()
  }
  match outcome {
    Err(error) => raise translate_borrow_failure(error)
    Ok(_) if !consumer.was_accepted() =>
      raise HostArenaNotProvided(arena_ordinal)
    Ok(_) => ()
  }
}

///|
/// Exercise every planned family-owned arena borrow without allocating or
/// copying device memory. This grants no materialization authority.
pub fn validate_borrowed_source(
  layout : SegmentedDeviceLayout,
  provide~ : (Int, @segmented_host_borrow.SegmentedHostArenaConsumer) -> Unit raise @segmented_host_borrow.SegmentedHostBorrowError,
) -> Unit raise SegmentedDeviceMaterializationError {
  if recompute_layout_digest(layout) != layout.digest {
    raise LayoutDigestMismatch
  }
  for region in layout.regions {
    provide_borrowed_region(layout, region, provide, _ => ())
  }
}

///|
fn[T] prepare_with_borrowed_resource(
  layout : SegmentedDeviceLayout,
  expected_identity : @spec.ModelIdentity,
  expected_source_digest : SegmentedHostSourceDigest,
  expected_layout_digest : SegmentedDeviceLayoutDigest,
  provide : (Int, @segmented_host_borrow.SegmentedHostArenaConsumer) -> Unit raise @segmented_host_borrow.SegmentedHostBorrowError,
  allocate : (Int64) -> T raise SegmentedDeviceMaterializationError,
  validate : (T, Int64, Int64, Int64) -> Unit raise SegmentedDeviceMaterializationError,
  copy : (T, FixedArray[Byte], Int64, Int64, Int64) -> Unit raise SegmentedDeviceMaterializationError,
  close : (T) -> Unit raise @device.DeviceError,
) -> SegmentedResourcePreparation[T] raise SegmentedDeviceMaterializationError {
  if layout.source_identity != expected_identity {
    raise IdentityMismatch
  }
  if layout.source_digest != expected_source_digest {
    raise SourceDigestMismatch
  }
  if layout.digest != expected_layout_digest ||
    recompute_layout_digest(layout) != layout.digest {
    raise LayoutDigestMismatch
  }
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
    let outcome : Result[Unit, SegmentedDeviceMaterializationError] = Ok(
      provide_borrowed_region(layout, region, provide, source => {
        copy(
          resource,
          source,
          region.source_offset.to_int64(),
          region.device_offset,
          region.byte_count,
        )
      }),
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

///|
/// Device materialization through synchronously borrowed opaque family-owned
/// host arenas. No mutable arena is returned through this API.
pub fn materialize_from_borrowed_source(
  context : @device.Context,
  layout : SegmentedDeviceLayout,
  expected_identity~ : @spec.ModelIdentity,
  expected_source_digest~ : SegmentedHostSourceDigest,
  expected_layout_digest~ : SegmentedDeviceLayoutDigest,
  provide~ : (Int, @segmented_host_borrow.SegmentedHostArenaConsumer) -> Unit raise @segmented_host_borrow.SegmentedHostBorrowError,
) -> SegmentedDevicePreparation raise SegmentedDeviceMaterializationError {
  let prepared = prepare_with_borrowed_resource(
    layout,
    expected_identity,
    expected_source_digest,
    expected_layout_digest,
    provide,
    bytes => {
      context.allocate(bytes) catch {
        error => raise DeviceFailed(error)
      }
    },
    (allocation, offset, byte_count, alignment) => {
      context.validate_allocation_region(
        allocation,
        offset~,
        byte_count~,
        alignment~,
      ) catch {
        error => raise DeviceFailed(error)
      }
    },
    (allocation, source, source_offset, destination_offset, byte_count) => {
      context.copy_from_fixed_host(
        allocation, source, source_offset, destination_offset, byte_count,
      ) catch {
        error => raise DeviceFailed(error)
      }
    },
    allocation => allocation.close(),
  )
  match prepared {
    SegmentedResourceReady(ready) =>
      SegmentedDeviceWeightsReady({
        allocation: ready.resource,
        layout: ready.layout,
      })
    SegmentedResourceCleanupRequired(failed) =>
      SegmentedDeviceCleanupRequired({
        allocation: failed.resource,
        cause: failed.cause,
        stage: failed.stage,
        cleanup_failure: failed.cleanup_failure,
        open: true,
      })
  }
}
