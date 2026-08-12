///|
priv struct FailedOwnedResource[Resource] {
  resource : Resource
  cause : DeviceKvArenaError
  cleanup_failure : DeviceKvArenaError
}

///|
priv enum OwnedResourcePreparation[Resource] {
  OwnedReady(Resource)
  OwnedCleanupRequired(FailedOwnedResource[Resource])
}

///|
fn[Resource] retain_after_validation_failure(
  resource : Resource,
  cause : DeviceKvArenaError,
  close : (Resource) -> Unit raise DeviceKvArenaError,
) -> OwnedResourcePreparation[Resource] raise DeviceKvArenaError {
  try close(resource) catch {
    cleanup_failure =>
      OwnedCleanupRequired({ resource, cause, cleanup_failure })
  } noraise {
    _ => raise cause
  }
}

///|
fn[Resource] prepare_owned_resource(
  expected_identity : @spec.ModelIdentity,
  layout : @device_layout.DeviceKvLayout,
  open : (Int64) -> Resource raise DeviceKvArenaError,
  byte_count : (Resource) -> Int64,
  validate : (Resource, Int64, Int64) -> Unit raise DeviceKvArenaError,
  close : (Resource) -> Unit raise DeviceKvArenaError,
) -> (DeviceKvArenaPlan, OwnedResourcePreparation[Resource]) raise DeviceKvArenaError {
  let plan = preflight(expected_identity, layout)
  let resource = open(plan.arena_bytes())
  if byte_count(resource) != plan.arena_bytes() {
    let failed = retain_after_validation_failure(
      resource,
      AllocationSizeMismatch,
      close,
    )
    return (plan, failed)
  }
  try
    validate(resource, plan.arena_bytes(), plan.segment_alignment_bytes())
  catch {
    cause => {
      let failed = retain_after_validation_failure(resource, cause, close)
      (plan, failed)
    }
  } noraise {
    _ => (plan, OwnedReady(resource))
  }
}

///|
/// Allocates exactly one persistent device allocation after semantic
/// preflight, then proves its exact byte count and physical base alignment.
///
/// The returned owner must be closed before `context`. A construction error is
/// raised after successful cleanup; `CleanupRequired` is returned only when
/// both validation and its first cleanup attempt fail.
pub fn prepare(
  context : @device.Context,
  expected_identity : @spec.ModelIdentity,
  layout : @device_layout.DeviceKvLayout,
) -> DeviceKvArenaPreparation raise DeviceKvArenaError {
  let (plan, outcome) = prepare_owned_resource(
    expected_identity,
    layout,
    bytes => {
      context.allocate(bytes) catch {
        error => raise DeviceFailed(AllocationCreate, error)
      }
    },
    allocation => allocation.byte_count(),
    (allocation, bytes, alignment) => {
      context.validate_allocation_region(
        allocation,
        offset=0L,
        byte_count=bytes,
        alignment~,
      ) catch {
        error => raise DeviceFailed(AllocationValidation, error)
      }
    },
    allocation => {
      allocation.close() catch {
        error => raise DeviceFailed(AllocationClose, error)
      }
    },
  )
  match outcome {
    OwnedReady(allocation) =>
      Ready({
        allocation,
        plan,
        context_ordinal: context.ordinal(),
        lifecycle: Open,
      })
    OwnedCleanupRequired(failed) =>
      CleanupRequired({
        allocation: failed.resource,
        cause: failed.cause,
        cleanup_failure: failed.cleanup_failure,
        lifecycle: ClosingFailed,
      })
  }
}
