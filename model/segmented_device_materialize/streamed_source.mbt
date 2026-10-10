///|
fn[T] prepare_with_streamed_resource(
  layout : SegmentedDeviceLayout,
  allocate : (Int64) -> T raise SegmentedDeviceMaterializationError,
  validate : (T, Int64, Int64, Int64) -> Unit raise SegmentedDeviceMaterializationError,
  upload : (T) -> Unit raise SegmentedDeviceMaterializationError,
  close : (T) -> Unit raise @device.DeviceError,
) -> SegmentedResourcePreparation[T] raise SegmentedDeviceMaterializationError {
  // Consume the validated immutable layout without serializing/hashing it
  // again. Allocation-region validation below protects the actual copy.
  let resource = allocate(layout.device_arena_bytes)
  for region in layout.regions {
    validate(
      resource,
      region.device_offset,
      region.byte_count,
      layout.alignment_bytes,
    ) catch {
      cause =>
        return failed_resource(
          resource,
          cause,
          AllocationRegionValidation,
          close,
        )
    }
  }
  upload(resource) catch {
    cause => return failed_resource(resource, cause, Transfer, close)
  }
  SegmentedResourceReady({ resource, layout, })
}

///|
/// Materialize the existing segmented allocation ABI through a synchronous
/// startup uploader. It can stream filesystem slices without constructing any
/// source arenas. Layout planning remains pure; only this boundary owns effects.
pub fn materialize_from_streamed_source(
  context : @device.Context,
  layout : SegmentedDeviceLayout,
  upload~ : (@device.Allocation) -> Unit raise SegmentedDeviceMaterializationError,
) -> SegmentedDevicePreparation raise SegmentedDeviceMaterializationError {
  let prepared = prepare_with_streamed_resource(
    layout,
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
    upload,
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
