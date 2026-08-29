///|
/// Builds one startup-only opaque kernel argument for an exact semantic KV
/// layer/component span. The arena retains the allocation and close authority;
/// no raw allocation or device pointer crosses this package boundary.
pub fn DeviceKvArena::component_argument(
  self : DeviceKvArena,
  context : @device.Context,
  layer : @plan.KvLayerId,
  component : @device_layout.DeviceKvComponent,
) -> @device.KernelArgument raise DeviceKvArenaError {
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
  @device.KernelArgument::device_region(
    self.allocation,
    offset=span.offset_bytes,
    byte_count=span.byte_count,
    alignment=span.alignment_bytes,
  ) catch {
    error => raise DeviceFailed(RegionValidation, error)
  }
}
