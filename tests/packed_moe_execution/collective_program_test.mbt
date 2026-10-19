///|
fn run_collective_program(
  queued : Bool,
  replay? : Bool = false,
  collective_graph? : Bool = false,
  bucketed? : Bool = false,
  decode_graph? : Bool = false,
  cancel? : Bool = false,
  cancel_rows? : Int = 1,
  fork? : Bool = false,
  abort_idle? : Bool = false,
  bfloat16? : Bool = false,
) -> Unit raise {
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
        kind=if bfloat16 { Bf16SumAllReduce } else { F32SumAllReduce },
        full_width=8,
        local_width=8,
      ),
      @device.TensorCollectiveSite::new(
        sequence=2,
        operation_id=8,
        kind=if bfloat16 { Bf16SumAllReduce } else { F32SumAllReduce },
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
        bfloat16~,
      ),
      @collective_program.ReductionBoundary::new(
        before_kernel=3,
        width=8,
        operation_id=8,
        buffer~,
        bfloat16~,
      ),
    ],
    owner~,
    generation=1UL,
    maximum_rows=2,
    retirement=if queued { WholeFrame } else { EachCollective },
    replay=if decode_graph {
      CaptureDecodeCollectives
    } else if bucketed {
      CaptureBucketedCollectives
    } else if collective_graph {
      CapturePaddedCollectives
    } else if replay {
      CaptureIntervals
    } else {
      Eager
    },
    replay_budget_bytes=1048576L,
    forks=if fork {
      [@parallel.ForkJoinSpan::new(first=1, split=2, join=3)]
    } else {
      []
    },
  )
  defer (queue.abort() catch { _ => () })
  assert_eq(
    queue.replay_graph_count(),
    if decode_graph {
      4
    } else if bucketed {
      2
    } else if collective_graph {
      1
    } else if replay {
      3
    } else {
      0
    },
  )
  let before = enqueues()
  let syncs = blocking_syncs()
  begin_allocations()
  for sequence in 1UL..<17UL {
    queue.submit(
      live_rows=if sequence % 2UL == 0UL { 2 } else { 1 },
      plan_sequence=sequence,
      global_sequence=2UL * sequence - 1UL,
    )
    if queue.poll() {
      abort("premature retirement")
    }
    if !collective_graph &&
      !bucketed &&
      !(decode_graph && sequence % 2UL != 0UL) &&
      enqueues() != before + (sequence - 1UL) * 5UL + 1UL {
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
  if abort_idle {
    queue.abort()
    owner.abort()
    queue.close()
    buffer.close()
    return
  }
  if cancel {
    queue.submit(
      live_rows=cancel_rows,
      plan_sequence=17UL,
      global_sequence=33UL,
    )
    ignore(queue.poll())
    try queue.close() catch {
      _ => ()
    } noraise {
      _ => fail("active queue closed")
    }
    try owner.close() catch {
      _ => ()
    } noraise {
      _ => fail("pending collective closed")
    }
    queue.abort()
    owner.abort()
    buffer.close()
    return
  }
  queue.close()
  owner.close()
}

///|
test "decoder queue splices multiple tensor reductions before consumers without allocation" {
  run_collective_program(false)
}

///|
test "independent producer submission and cancellation retain both lanes without allocation" {
  run_collective_program(true, fork=true)
  run_collective_program(true, fork=true, cancel=true)
}

///|
test "queued decoder retains operands through one terminal retirement without allocation" {
  run_collective_program(true)
}

///|
test "captured intervals preserve collective cuts and replay with zero allocation" {
  run_collective_program(true, replay=true)
}

///|
test "queued and captured cancellation retain live work until explicit abort" {
  run_collective_program(true, cancel=true)
  run_collective_program(true, replay=true, cancel=true)
  run_collective_program(true, collective_graph=true, cancel=true)
  run_collective_program(true, bucketed=true, cancel=true)
  run_collective_program(true, decode_graph=true, cancel=true)
  run_collective_program(true, decode_graph=true, cancel=true, cancel_rows=2)
}

///|
test "collective-inclusive replay retires entire effect frame without allocation" {
  run_collective_program(true, collective_graph=true)
  run_collective_program(true, bucketed=true)
  run_collective_program(true, decode_graph=true)
}

///|
test "idle peer teardown drops retained collective graphs before abort" {
  run_collective_program(true, collective_graph=true, abort_idle=true)
  run_collective_program(true, bucketed=true, abort_idle=true)
  run_collective_program(true, decode_graph=true, abort_idle=true)
}

///|
test "BF16 local-combine boundaries preserve eager replay forks and cancellation" {
  run_collective_program(false, bfloat16=true)
  run_collective_program(true, bfloat16=true, fork=true)
  run_collective_program(true, bfloat16=true, replay=true)
  run_collective_program(true, bfloat16=true, decode_graph=true)
  run_collective_program(true, bfloat16=true, decode_graph=true, cancel=true)
}
