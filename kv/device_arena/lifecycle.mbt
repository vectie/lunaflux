///|
fn require_open(
  lifecycle : DeviceKvArenaLifecycle,
) -> Unit raise DeviceKvArenaError {
  if lifecycle != Open {
    raise Closed
  }
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
/// Resolves and physically validates one exact page component without exposing
/// the allocation. This is a preparation/dispatch guard, not mutation
/// authority. A context with a different ordinal is rejected before touching
/// the device facade; same-ordinal alias mistakes are rejected by the facade's
/// native allocation/context validation.
pub fn DeviceKvArena::validate_region(
  self : DeviceKvArena,
  context : @device.Context,
  layer_index : Int,
  component : @device_layout.DeviceKvComponent,
  page_index : Int,
) -> @device_layout.DeviceKvRegion raise DeviceKvArenaError {
  validate_owned_region(
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
}

///|
/// Creates one safe opaque kernel argument for the complete persistent arena.
/// Paged kernels bind this argument once and resolve page offsets from the
/// immutable layout and authenticated host page indices. The returned value is
/// a borrow: it must not be launched after this arena begins closing.
pub fn DeviceKvArena::kernel_argument(
  self : DeviceKvArena,
  context : @device.Context,
) -> @device.KernelArgument raise DeviceKvArenaError {
  require_open(self.lifecycle)
  if context.ordinal() != self.context_ordinal {
    raise WrongContext
  }
  context.validate_allocation_region(
    self.allocation,
    offset=0L,
    byte_count=self.plan.arena_bytes(),
    alignment=self.plan.segment_alignment_bytes(),
  ) catch {
    error => raise DeviceFailed(KernelBinding, error)
  }
  @device.KernelArgument::device_region(
    self.allocation,
    offset=0L,
    byte_count=self.plan.arena_bytes(),
    alignment=self.plan.segment_alignment_bytes(),
  ) catch {
    error => raise DeviceFailed(KernelBinding, error)
  }
}

///|
/// Creates a safe opaque kernel argument for one exact canonical page segment.
/// Indices are resolved against the retained layout so a region from another
/// arena cannot be confused with this allocation.
pub fn DeviceKvArena::region_kernel_argument(
  self : DeviceKvArena,
  context : @device.Context,
  layer_index : Int,
  component : @device_layout.DeviceKvComponent,
  page_index : Int,
) -> @device.KernelArgument raise DeviceKvArenaError {
  let region = self.validate_region(context, layer_index, component, page_index)
  @device.KernelArgument::device_region(
    self.allocation,
    offset=region.offset_bytes(),
    byte_count=region.reserved_bytes(),
    alignment=region.alignment_bytes(),
  ) catch {
    error => raise DeviceFailed(KernelBinding, error)
  }
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
