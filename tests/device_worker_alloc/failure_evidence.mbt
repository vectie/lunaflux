///|
extern "C" fn fake_set_fault(mode : Int) = "lunaflux_device_worker_fake_set_fault"

///|
extern "C" fn fake_fault_mode() -> Int = "lunaflux_device_worker_fake_fault_mode"

///|
fn assert_execute_failure(
  directory : String,
  manifest : String,
  model : @spec.LlamaModelMetadata,
  model_plan : @plan.ModelPlan,
  target : @catalog.DeviceTarget,
  fault_mode : Int,
  branch : Int,
) -> Unit raise {
  let owner = prepare_fixture_owner(
    directory, manifest, model, model_plan, target,
  )
  let allocator = @page_allocator.PageAllocator::new(
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=1,
      max_run_pages=1,
      max_active_references_per_page=1,
      max_cached_references_per_page=1,
    ),
  )
  let page = allocator.allocate_one()
  let cycle = prepare_cycle_for_branch(1UL, branch, model_plan, page)
  let completions = @worker_wire.CompletionFrameBuffer::new(
    fixture_worker_limits(),
    fixture_inference_limits(),
  )
  let writer = completions.writer_for(cycle.frame)
  let copies = fake_copy_count()
  let launches = fake_launch_count()
  let syncs = fake_sync_count()
  let reads = fake_readback_count()
  let creates = fake_resource_create_count()
  let closes = fake_resource_close_count()
  fake_set_fault(fault_mode)
  try owner.execute_frame(cycle.frame, writer) catch {
    @device_worker.Executor(
      @device_step.ExecutorLaunchFailed(
        _,
        @device.DriverFailure(@device.AotKernelLaunch)
      )
    ) if fault_mode == 1 => ()
    @device_worker.Executor(
      @device_step.ExecutorDevice(
        @device_step.ActivationReadback,
        @device.DriverFailure(@device.CopyToHost)
      )
    ) if fault_mode == 2 => ()
    @device_worker.Executor(
      @device_step.ExecutorSampling(@sampling.NonFiniteLogit(row=0, token_id=0))
    ) if fault_mode == 3 => ()
    _ => alloc_probe_abort()
  } noraise {
    _ => alloc_probe_abort()
  }
  if fake_fault_mode() != 0 {
    alloc_probe_abort()
  }
  let expected_launches = if fault_mode == 1 {
    0UL
  } else {
    model_plan.operations().length().to_uint64()
  }
  let expected_reads = if fault_mode == 3 { 1UL } else { 0UL }
  if fake_copy_count() - copies != 8UL ||
    fake_launch_count() - launches != expected_launches ||
    fake_sync_count() - syncs != expected_launches ||
    fake_readback_count() - reads != expected_reads ||
    fake_resource_create_count() != creates ||
    fake_resource_close_count() != closes {
    alloc_probe_abort()
  }
  if owner.lifecycle() != WorkerFaulted {
    alloc_probe_abort()
  }
  try completions.current() catch {
    StaleFrame => ()
    _ => alloc_probe_abort()
  } noraise {
    _ => alloc_probe_abort()
  }
  let reuse = completions.writer_for(cycle.frame)
  reuse.abort()
  owner.close()
  allocator.release_active(page)
  if fake_live_children() != 0 ||
    allocator.allocated_page_count() != 0 ||
    allocator.free_page_count() != 1 {
    alloc_probe_abort()
  }
}

///|
fn run_failure_evidence(
  directory : String,
  manifest : String,
  model : @spec.LlamaModelMetadata,
  model_plan : @plan.ModelPlan,
  target : @catalog.DeviceTarget,
) -> Unit raise {
  // Launch, readback, and non-finite BF16 output faults all enter the real
  // aggregate execute path. Finish ordering is separately covered by the
  // device_worker package's package-private transaction test.
  assert_execute_failure(
    directory,
    manifest,
    model,
    model_plan,
    target,
    1,
    CYCLE_ORDINARY_PREFILL,
  )
  assert_execute_failure(
    directory,
    manifest,
    model,
    model_plan,
    target,
    2,
    CYCLE_GREEDY_DECODE,
  )
  assert_execute_failure(
    directory,
    manifest,
    model,
    model_plan,
    target,
    3,
    CYCLE_STOCHASTIC_DECODE,
  )
}
