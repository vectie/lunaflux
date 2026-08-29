///|
fn require_open(
  lifecycle : DeviceKvArenaLifecycle,
) -> Unit raise DeviceKvArenaError {
  match lifecycle {
    Open => ()
    PoisonedState => raise ArenaPoisoned
    ClosingFailed | ClosedState => raise Closed
  }
}

///|
fn poison_lifecycle(
  lifecycle : DeviceKvArenaLifecycle,
) -> DeviceKvArenaLifecycle {
  if lifecycle == Open {
    PoisonedState
  } else {
    lifecycle
  }
}

///|
/// Permanently disables future region validation after a dispatch failure.
/// Poisoning is idempotent and preserves the owner's explicit close authority.
pub fn DeviceKvArena::poison(self : DeviceKvArena) -> Unit {
  self.lifecycle = poison_lifecycle(self.lifecycle)
}

///|
fn[Resource] close_retry_safe(
  lifecycle : DeviceKvArenaLifecycle,
  set_lifecycle : (DeviceKvArenaLifecycle) -> Unit,
  resource : Resource,
  close : (Resource) -> Unit raise DeviceKvArenaError,
) -> Unit raise DeviceKvArenaError {
  if lifecycle == ClosedState {
    return
  }
  set_lifecycle(ClosingFailed)
  close(resource)
  set_lifecycle(ClosedState)
}

///|
fn[Resource, ContextIdentity : Eq] validate_owned_region(
  lifecycle : DeviceKvArenaLifecycle,
  resource : Resource,
  layout : @device_layout.DeviceKvLayout,
  owning_context : ContextIdentity,
  supplied_context : ContextIdentity,
  layer_index : Int,
  component : @device_layout.DeviceKvComponent,
  page_index : Int,
  validate : (Resource, Int64, Int64, Int64) -> Unit raise DeviceKvArenaError,
) -> @device_layout.DeviceKvRegion raise DeviceKvArenaError {
  require_open(lifecycle)
  if supplied_context != owning_context {
    raise WrongContext
  }
  let region = layout.region(layer_index, component, page_index) catch {
    _ => raise InvalidLayout(RegionIndex)
  }
  validate(
    resource,
    region.offset_bytes(),
    region.reserved_bytes(),
    region.alignment_bytes(),
  )
  region
}

///|
/// Exact private layer/component span retained for package-local argument
/// construction. The public bridge exposes only an opaque kernel argument;
/// allocation and close authority remain here.
#valtype
priv struct DeviceKvComponentSpan {
  offset_bytes : Int64
  byte_count : Int64
  alignment_bytes : Int64
}

///|
fn[Resource, ContextIdentity : Eq] validate_owned_component_span(
  lifecycle : DeviceKvArenaLifecycle,
  resource : Resource,
  layout : @device_layout.DeviceKvLayout,
  owning_context : ContextIdentity,
  supplied_context : ContextIdentity,
  layer : @plan.KvLayerId,
  component : @device_layout.DeviceKvComponent,
  validate : (Resource, Int64, Int64, Int64) -> Unit raise DeviceKvArenaError,
) -> DeviceKvComponentSpan raise DeviceKvArenaError {
  require_open(lifecycle)
  if supplied_context != owning_context {
    raise WrongContext
  }
  let first = layout.region(layer.as_int(), component, 0) catch {
    _ => raise InvalidLayout(RegionIndex)
  }
  let bytes = layout.component_stride_bytes()
  let alignment = layout.segment_alignment_bytes()
  validate(resource, first.offset_bytes(), bytes, alignment)
  {
    offset_bytes: first.offset_bytes(),
    byte_count: bytes,
    alignment_bytes: alignment,
  }
}

///|
/// Resolves and physically validates one exact page component without exposing
/// the allocation. This startup/diagnostic guard first proves the containing
/// semantic-layer component span and then the exact page; it is not a token-
/// step dispatch path or mutation authority. A context with a different
/// ordinal is rejected before touching the device facade; same-ordinal alias
/// mistakes are rejected by the facade's native allocation/context validation.
pub fn DeviceKvArena::validate_region(
  self : DeviceKvArena,
  context : @device.Context,
  layer_index : Int,
  component : @device_layout.DeviceKvComponent,
  page_index : Int,
) -> @device_layout.DeviceKvRegion raise DeviceKvArenaError {
  let layer = @plan.KvLayerId::new(layer_index) catch {
    _ => raise InvalidLayout(RegionIndex)
  }
  let span = validate_owned_component_span(
    self.lifecycle,
    self.allocation,
    self.plan.layout(),
    self.context_ordinal,
    context.ordinal(),
    layer,
    component,
    (allocation, offset, bytes, alignment) => {
      context.validate_allocation_region(
        allocation,
        offset~,
        byte_count=bytes,
        alignment~,
      ) catch {
        error => raise DeviceFailed(RegionValidation, error)
      }
    },
  )
  let region = validate_owned_region(
    self.lifecycle,
    self.allocation,
    self.plan.layout(),
    self.context_ordinal,
    context.ordinal(),
    layer_index,
    component,
    page_index,
    (allocation, offset, bytes, alignment) => {
      context.validate_allocation_region(
        allocation,
        offset~,
        byte_count=bytes,
        alignment~,
      ) catch {
        error => raise DeviceFailed(RegionValidation, error)
      }
    },
  )
  if region.offset_bytes() < span.offset_bytes ||
    region.end_offset_bytes() > span.offset_bytes + span.byte_count ||
    region.alignment_bytes() != span.alignment_bytes {
    raise InvalidLayout(RegionIndex)
  }
  region
}

///|
/// Closes the allocation deterministically. Success is idempotent. A failed
/// close makes the arena unusable but retains retry authority in this owner.
pub fn DeviceKvArena::close(
  self : DeviceKvArena,
) -> Unit raise DeviceKvArenaError {
  close_retry_safe(
    self.lifecycle,
    lifecycle => self.lifecycle = lifecycle,
    self.allocation,
    allocation => {
      allocation.close() catch {
        error => raise DeviceFailed(AllocationClose, error)
      }
    },
  )
}

///|
/// Retries release of the allocation retained after failed construction.
/// Success is idempotent; failure updates the inspectable cleanup cause while
/// preserving authority for another retry.
pub fn FailedDeviceKvArenaPreparation::retry_close(
  self : FailedDeviceKvArenaPreparation,
) -> Unit raise DeviceKvArenaError {
  close_retry_safe(
    self.lifecycle,
    lifecycle => self.lifecycle = lifecycle,
    self.allocation,
    allocation => {
      allocation.close() catch {
        error => {
          let translated = DeviceFailed(AllocationClose, error)
          self.cleanup_failure = translated
          raise translated
        }
      }
    },
  )
}
