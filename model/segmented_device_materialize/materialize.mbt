///|
/// Allocate one device arena and synchronously copy every ordered source
/// region. Caller-owned host arenas are borrowed only for each copy call.
pub fn materialize(
  context : @device.Context,
  layout : SegmentedDeviceLayout,
  source_arenas~ : ArrayView[FixedArray[Byte]],
  expected_identity~ : @spec.ModelIdentity,
  expected_source_digest~ : SegmentedHostSourceDigest,
  expected_layout_digest~ : SegmentedDeviceLayoutDigest,
) -> SegmentedDevicePreparation raise SegmentedDeviceMaterializationError {
  let prepared = prepare_with_resource(
    layout,
    source_arenas,
    expected_identity,
    expected_source_digest,
    expected_layout_digest,
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
