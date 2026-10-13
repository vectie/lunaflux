///|
fn run_collective_program(queued : Bool) -> Unit raise {
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let buffer = context.allocate(64L)
  defer (buffer.close() catch { _ => () })
  let module = context.load_module(b"fake\x00")
  defer (module.close() catch { _ => () })
  let function = module.load_function(b"moe\x00")
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
        operation_id=3,
        kind=F32SumAllReduce,
        full_width=8,
        local_width=8,
      ),
      @device.TensorCollectiveSite::new(
        sequence=2,
        operation_id=8,
        kind=F32SumAllReduce,
        full_width=8,
        local_width=8,
      ),
    ],
  )
  let owner = context.create_tensor_collective_owner(
    runtime~,
    group_id=runtime.create_group_id(),
    contract~,
    generation=1UL,
    previous_plan_sequence_value=0UL,
  )
  defer (owner.abort() catch { _ => () })
  while !owner.poll_ready() {
    ()
  }
  owner.validate_group(rank=0, world_size=2)
  try owner.validate_group(rank=1, world_size=2) catch {
    _ => ()
  } noraise {
    _ => fail("wrong rank accepted")
  }
  try
    owner.validate_sum_site(
      site_sequence=1,
      operation_id=3,
      maximum_rows=2,
      width=4,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("wrong reduction width accepted")
  }
  let queue = @collective_program.CollectiveKernelProgram::new(
    context,
    stream,
    [launch, launch, launch, launch, launch],
    [
      @collective_program.ReductionBoundary::new(
        before_kernel=1,
        width=8,
        operation_id=3,
        buffer~,
      ),
      @collective_program.ReductionBoundary::new(
        before_kernel=3,
        width=8,
        operation_id=8,
        buffer~,
      ),
    ],
    owner~,
    generation=1UL,
    maximum_rows=2,
    retirement=if queued { WholeFrame } else { EachCollective },
  )
  defer (queue.abort() catch { _ => () })
  let before = enqueues()
  let syncs = blocking_syncs()
  begin_allocations()
  for sequence in 1UL..<17UL {
    queue.submit(
      live_rows=2,
      plan_sequence=sequence,
      global_sequence=2UL * sequence - 1UL,
    )
    if queue.poll() {
      abort("premature retirement")
    }
    if enqueues() != before + (sequence - 1UL) * 5UL + 1UL {
      abort("consumer submitted before first reduction")
    }
    while !queue.poll() {
      ()
    }
  }
  let allocations = end_allocations()
  assert_eq(allocations, 0UL)
  assert_eq(enqueues() - before, 80UL)
  assert_eq(blocking_syncs() - syncs, 0UL)
  queue.close()
  owner.close()
}

///|
test "decoder queue splices multiple tensor reductions before consumers without allocation" {
  run_collective_program(false)
}

///|
test "queued decoder retains operands through one terminal retirement without allocation" {
  run_collective_program(true)
}
