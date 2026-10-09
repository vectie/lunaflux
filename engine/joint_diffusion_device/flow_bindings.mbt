///|
/// Borrowed startup-uploaded allocation. The caller owns this allocation and
/// must leave it immutable and alive until all frame queues have released it.
pub struct FlowBindings {
  priv table : @joint_diffusion_loop.FlowTable
  priv allocation : @device.Allocation
  priv base : Int64
}

///|
/// Validate once and upload the full table once. Allocation failure/cleanup is
/// caller-owned; this routine creates no native resource that can be lost if
/// upload fails. Per-step arguments below acquire queue leases at frame setup.
pub fn FlowBindings::upload(
  context : @device.Context,
  table : @joint_diffusion_loop.FlowTable,
  allocation : @device.Allocation,
  offset~ : Int64,
) -> Self raise @device.DeviceError {
  context.validate_allocation_region(
    allocation,
    offset~,
    byte_count=table.byte_count(),
    alignment=16L,
  )
  allocation.copy_from_host(table.bytes(), 0L, offset, table.byte_count())
  { table, allocation, base: offset, }
}

///|
/// Create a fixed 12-byte RF coefficient argument during frame preparation.
pub fn FlowBindings::coefficients(
  self : Self,
  step : Int,
  latent : @joint_diffusion_plan.JointDiffusionLatentKind,
) -> @device.KernelArgument raise {
  let offset = self.base + self.table.coefficient_offset(step, latent)
  @device.KernelArgument::device_region(
    self.allocation,
    offset~,
    byte_count=12L,
    alignment=16L,
  )
}

///|
/// Scalar normalized time, for preparation paths accepting a single F32 cell.
/// Packed distinct timestep tables (including conditioning rows) are separate.
pub fn FlowBindings::timestep(
  self : Self,
  step : Int,
  latent : @joint_diffusion_plan.JointDiffusionLatentKind,
) -> @device.KernelArgument raise {
  let offset = self.base + self.table.coefficient_offset(step, latent) + 12L
  @device.KernelArgument::device_region(
    self.allocation,
    offset~,
    byte_count=4L,
    alignment=4L,
  )
}
