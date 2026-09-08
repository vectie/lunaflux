///|
/// Startup-only assembly of the second map/reduce stage. The scratch is owned
/// by the existing deterministic sampling resource lifecycle.
fn prepare_paged_segmented_greedy_merge(
  operation_id : @plan.OperationId,
  limits : DeviceStepLimits,
  admission : PagedCudaGreedySamplingAdmission,
  descriptor : DeviceStepResources,
  resources : PagedGraphResources,
) -> PreparedPagedGraphStep raise DeviceStepError {
  guard resources.modules.get(admission.module_index) is Some(module_owner) &&
    module_owner.open &&
    resources.device_sampling_workspace is Some(workspace) &&
    workspace.open &&
    resources.greedy_sampling_results is Some(results) &&
    results.open else {
    raise InvalidExecutor(SamplingArtifact)
  }
  let function_resource = module_owner.resource.load_function(
    @utf8.encode(@luna_cuda_sampling_aot.segmented_greedy_merge_symbol()),
  ) catch {
    error => raise ExecutorDevice(Function, error)
  }
  let function_index = resources.functions.length()
  resources.functions.push({
    resource: function_resource,
    module_index: admission.module_index,
    open: true,
  })
  let arguments = @device.KernelArguments::new([
    @device.KernelArgument::device_region(
      descriptor.counts,
      offset=if sampling_descriptor_is_projected(resources) {
        byte_capacity(OUTPUT_COUNTS_CELL)
      } else {
        0L
      },
      byte_count=16L,
      alignment=4L,
    ),
    @device.KernelArgument::device_region(
      workspace.resource,
      offset=0L,
      byte_count=workspace.resource.byte_count(),
      alignment=4L,
    ),
    @device.KernelArgument::device_region(
      results.resource,
      offset=0L,
      byte_count=results.resource.byte_count(),
      alignment=4L,
    ),
  ]) catch {
    error => raise ExecutorDevice(KernelArguments, error)
  }
  let dimensions = @device.KernelLaunchDimensions::new(
    grid_x=limits.max_rows(),
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    _ => raise InvalidExecutor(LaunchDimensions)
  }
  { operation_id, function_index, dimensions, arguments }
}
