///|
extern "C" fn begin_allocations() = "lunaflux_device_worker_alloc_probe_begin"

///|
extern "C" fn end_allocations() -> UInt64 = "lunaflux_device_worker_alloc_probe_end"

///|
extern "C" fn enqueues() -> UInt64 = "lunaflux_tp_alloc_executor_enqueues"

///|
extern "C" fn blocking_syncs() -> UInt64 = "lunaflux_tp_alloc_blocking_synchronizations"

///|
test "prepared MoE inserts F32 reduction, reuses its queue and allocates nothing" {
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let buffer = context.allocate(64L)
  defer (buffer.close() catch { _ => () })
  let kernel_module = context.load_module(b"fake\x00")
  defer (kernel_module.close() catch { _ => () })
  let function = kernel_module.load_function(b"moe\x00")
  defer (function.close() catch { _ => () })
  let launch = @device.OrderedKernelLaunch::new(
    function,
    @device.KernelLaunchDimensions::new(
      grid_x=1,
      grid_y=1,
      grid_z=1,
      block_x=32,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    @device.KernelArguments::new([
      @device.KernelArgument::device_region(
        buffer,
        offset=0L,
        byte_count=64L,
        alignment=4L,
      ),
    ]),
  )
  let runtime = @device.admit_tensor_collective_runtime(
    minimum_version_code=21403,
    maximum_version_code=30000,
  )
  let contract = @device.TensorCollectiveContract::new(
    digest=@device.TensorCollectiveContractDigest::from_sha256(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    ),
    rank=0,
    world_size=2,
    max_query_tokens=2,
    sites=[
      @device.TensorCollectiveSite::new(
        sequence=1,
        operation_id=0,
        kind=F32SumAllReduce,
        full_width=8,
        local_width=8,
      ),
    ],
  )
  let collective = context.create_tensor_collective_owner(
    runtime~,
    group_id=runtime.create_group_id(),
    contract~,
    generation=1UL,
    previous_plan_sequence_value=0UL,
  )
  defer (collective.abort() catch { _ => () })
  while !collective.poll_ready() {
    ()
  }
  let executor = @moe.prepare_moe(
    context,
    stream,
    [launch],
    [launch],
    launch,
    routed_output=buffer,
    maximum_rows=2,
    reduction=@moe.MoeReduction::new(
      collective,
      generation=1UL,
      site_sequence=1,
      operation_id=0,
    ),
  )
  defer (executor.abort() catch { _ => () })
  let before = enqueues()
  begin_allocations()
  for sequence in 1UL..<17UL {
    executor.submit(
      live_rows=2,
      plan_sequence=sequence,
      global_sequence=sequence,
    )
    // Nonblocking communicator submission is pending on its first poll.
    if executor.poll() {
      abort("premature completion")
    }
    if enqueues() != before + (sequence - 1UL) * 3UL + 1UL {
      abort("shared work was submitted before the reduction")
    }
    while !executor.poll() {
      ()
    }
  }
  let allocations = end_allocations()
  assert_eq(allocations, 0UL)
  assert_eq(enqueues() - before, 48UL)
  assert_eq(blocking_syncs(), 0UL)
  executor.close()
  collective.close()
  function.close()
  kernel_module.close()
  buffer.close()
  stream.close()
  context.close()
}

///|
test "single-rank MoE reuses the same queue and can cancel queued work" {
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let buffer = context.allocate(64L)
  defer (buffer.close() catch { _ => () })
  let kernel_module = context.load_module(b"fake\x00")
  defer (kernel_module.close() catch { _ => () })
  let function = kernel_module.load_function(b"moe\x00")
  defer (function.close() catch { _ => () })
  let launch = @device.OrderedKernelLaunch::new(
    function,
    @device.KernelLaunchDimensions::new(
      grid_x=1,
      grid_y=1,
      grid_z=1,
      block_x=32,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    @device.KernelArguments::new([
      @device.KernelArgument::device_region(
        buffer,
        offset=0L,
        byte_count=64L,
        alignment=4L,
      ),
    ]),
  )
  let executor = @moe.prepare_moe(
    context,
    stream,
    [launch],
    [launch],
    launch,
    routed_output=buffer,
    maximum_rows=2,
  )
  defer (executor.abort() catch { _ => () })
  let before = enqueues()
  begin_allocations()
  for sequence in 1UL..<17UL {
    executor.submit(
      live_rows=1,
      plan_sequence=sequence,
      global_sequence=sequence,
    )
    while !executor.poll() {
      ()
    }
  }
  let allocations = end_allocations()
  assert_eq(allocations, 0UL)
  assert_eq(enqueues() - before, 48UL)
  executor.submit(live_rows=2, plan_sequence=17UL, global_sequence=17UL)
  executor.abort()
  executor.close()
}
