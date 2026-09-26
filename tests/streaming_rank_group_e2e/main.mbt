///|
async fn exchange(
  owner : @rank_group_process.RankGroupProcessSupervisor,
  command : @streaming_ir.Command,
) -> @streaming_ir.Reply {
  owner.begin_streaming_exchange(command)
  for _ in 0..<10000 {
    if owner.progress_streaming_exchange() is Some(reply) {
      return reply
    }
    @async.sleep(1)
  }
  fail("rank exchange timed out")
}

///|
async fn run_test() -> Unit {
  @async.with_task_group() <| group => {
    let args = @env.args()
    guard args.length() == 2 || (args.length() == 3 && args[2] == "--failure") else {
      fail(
        "expected absolute streaming_rank_echo executable and optional --failure",
      )
    }
    let executable = @worker_executable_fixture.admit(@utf8.encode(args[1]))
    group.add_defer(() => executable.close())
    let root = @approved_fs.ApprovedRoot::open_absolute("/tmp")
    group.add_defer(() => root.close())
    let roots = @approved_fs.acquire_worker_approved_roots(
      model_root=root,
      kernel_root=root,
    )
    group.add_defer(() => @approved_fs.close_worker_approved_roots(roots))
    let limits = ownership_worker_limits()
    let a = @worker_protocol.SchedulePlanBuffer::new(limits)
    let b = @worker_protocol.SchedulePlanBuffer::new(limits)
    let prepared = @rank_group_process.prepare_rank_group_process_approved(
      executable,
      [
        @rank_group_process.RankGroupProcessRankStartup::new(
          ownership_configure(0),
        ),
        @rank_group_process.RankGroupProcessRankStartup::new(
          ownership_configure(1),
        ),
      ],
      [a.identity(), b.identity()],
      previous_plan_sequence_value=0UL,
      @rank_group_process.RankGroupProcessLimits::new(
        max_payload_bytes=38912,
        startup_io_timeout_millis=10000,
        io_timeout_millis=10000,
        shutdown_timeout_millis=10000,
        collective_exchange_timeout_millis=10000,
      ),
    )
    let outcome = prepared.spawn_with_approved_root_spawn_authority(
      @approved_fs.worker_approved_root_spawn_authority(roots),
    )
    let owner = match outcome {
      ReadyRankGroupProcess(owner) => owner
      RankGroupProcessCleanupRequired(owner) => {
        owner.begin_cleanup()
        while owner.lifecycle() != Closed {
          ignore(owner.progress_cleanup())
          @async.sleep(1)
        }
        fail("rank startup failed")
      }
    }
    group.add_defer(() => {
      @async.protect_from_cancel(() => {
        if owner.lifecycle() != Closed {
          owner.fail_streaming()
          if owner.lifecycle() == Faulted {
            owner.begin_cleanup()
          }
          while owner.lifecycle() != Closed {
            ignore(owner.progress_cleanup())
            @async.sleep(1)
          }
        }
      })
    })
    let spill : @streaming_ir.Command = {
      worker: 1UL,
      sequence: 1UL,
      opcode: Spill,
      page: 0,
      page_generation: 1,
      host: { slot: -1, generation: 0, },
    }
    assert_true(exchange(owner, spill).status is Pending)
    let poll = { ..spill, opcode: Poll, }
    assert_true(exchange(owner, poll).status is Pending)
    assert_true(exchange(owner, poll).status is Pending)
    let spilled = exchange(owner, poll)
    assert_true(spilled.status is HostReady)
    let restore = {
      ..spill,
      sequence: 2UL,
      opcode: Restore,
      page: 1,
      host: spilled.host,
    }
    assert_true(exchange(owner, restore).status is Pending)
    assert_true(exchange(owner, { ..restore, opcode: Poll, }).status is Pending)
    // Rank zero has already completed; rank one must cancel before publication.
    assert_true(
      exchange(owner, { ..restore, opcode: Cancel, }).status is Cancelled,
    )
    assert_false(owner.streaming_busy())
    let restore = { ..restore, sequence: 3UL, }
    assert_true(exchange(owner, restore).status is Pending)
    for _ in 0..<2 {
      assert_true(
        exchange(owner, { ..restore, opcode: Poll, }).status is Pending,
      )
    }
    assert_true(
      exchange(owner, { ..restore, opcode: Poll, }).status is DeviceReady,
    )
    assert_true(
      exchange(owner, { ..restore, sequence: 4UL, opcode: Evict, }).status
      is Evicted,
    )
    if args.length() == 3 {
      let interrupted = { ..spill, sequence: 5UL, }
      assert_true(exchange(owner, interrupted).status is Pending)
      owner.begin_streaming_exchange({ ..interrupted, opcode: Poll, })
      ignore(owner.progress_streaming_exchange())
      owner.fail_streaming()
      assert_eq(owner.lifecycle(), Faulted)
      assert_eq(owner.inspect_group_failure().plan_sequence_value(), 0UL)
      assert_true(owner.streaming_busy())
      owner.begin_cleanup()
    } else {
      owner.begin_close()
    }
    while owner.lifecycle() != Closed {
      if owner.lifecycle() == Cleaning || owner.lifecycle() == Faulted {
        ignore(owner.progress_cleanup())
      } else {
        ignore(owner.progress_close())
      }
      @async.sleep(1)
    }
    println(
      if args.length() == 3 {
        "streaming rank group: active-transfer failure, cleanup and reap passed"
      } else {
        "streaming rank group: skewed completion, cancellation, restore, eviction and close passed"
      },
    )
  }
}

///|
async fn main {
  run_test() catch {
    @rank_group_process.Invalid(rule) =>
      fail("rank process invalid: \{Repr(rule)}")
    @rank_group_process.Failed(failure) =>
      fail("rank process failed: \{Repr(failure)}")
    error => raise error
  }
}
