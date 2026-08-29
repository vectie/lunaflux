///|
fn terminal_live_children(
  supervisor : RankGroupProcessSupervisor,
) -> Unit raise {
  let limits = @process.ProcessLimits::new(
    max_frame_bytes=4096,
    io_timeout_millis=1000,
    shutdown_timeout_millis=10,
  )
  for rank in 0..<supervisor.children.length() {
    supervisor.children[rank] = LiveChild(
      spawn_rank_fixture_child(@worker_executable_fixture.cat_path(), limits),
    )
  }
}

///|
fn terminal_finish_write(
  supervisor : RankGroupProcessSupervisor,
  rank : Int,
) -> Unit raise {
  for _ in 0..<100000 {
    if supervisor.children[rank].progress_write() {
      return
    }
  }
  fail("terminal write did not finish")
}

///|
fn terminal_response(
  supervisor : RankGroupProcessSupervisor,
  rank : Int,
  kind : @rank_group_wire.RankGroupWireKind,
) -> Int raise {
  @rank_group_wire.encode_rank_group_wire_control_into(
    supervisor.rx,
    supervisor.wire_limits,
    ownership_binding(rank),
    kind,
    ChildToParent,
    None,
    0UL,
  )
}

///|
fn terminal_reap(supervisor : RankGroupProcessSupervisor) -> Unit raise {
  for _ in 0..<100000 {
    match supervisor.progress_cleanup() {
      CleanupComplete => return
      CleanupStuck(rank~) => fail("terminal cleanup stuck at rank \{rank}")
      _ => ()
    }
  }
  fail("terminal cleanup did not finish")
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "healthy close drains and closes every rank before child reap" {
  let PreparedRankGroupProcess(supervisor) = ownership_prepared()
  terminal_live_children(supervisor)
  ownership_prime_wire_ready(supervisor)
  supervisor.lifecycle_value = Ready
  supervisor.begin_close()
  assert_eq(supervisor.lifecycle(), Draining)
  for rank in 0..<2 {
    terminal_finish_write(supervisor, rank)
    let length = terminal_response(supervisor, rank, Drained)
    supervisor.accept_terminal_response(rank, length)
    if rank == 0 {
      assert_false(supervisor.cleanup_started[0])
      assert_false(supervisor.cleanup_started[1])
    }
  }
  assert_eq(supervisor.lifecycle(), Closing)
  for rank in 0..<2 {
    terminal_finish_write(supervisor, rank)
    let length = terminal_response(supervisor, rank, Closed)
    supervisor.accept_terminal_response(rank, length)
  }
  assert_eq(supervisor.protocol.lifecycle(), Closed)
  assert_eq(supervisor.lifecycle(), Cleaning)
  assert_true(supervisor.cleanup_started[0])
  assert_true(supervisor.cleanup_started[1])
  terminal_reap(supervisor)
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "lost rank is force-reaped while survivor drains and closes" {
  let PreparedRankGroupProcess(supervisor) = ownership_prepared()
  terminal_live_children(supervisor)
  ownership_prime_wire_ready(supervisor)
  supervisor.lifecycle_value = Ready
  supervisor.protocol.report_idle_rank_loss(0)
  supervisor.record_failure(TransportLost(rank=0))
  assert_eq(supervisor.lifecycle(), Faulted)
  supervisor.begin_cleanup()
  assert_true(supervisor.survivor_terminal)
  assert_eq(supervisor.survivor_rank, 1)

  terminal_finish_write(supervisor, 1)
  let drained = terminal_response(supervisor, 1, Drained)
  supervisor.accept_survivor_response(1, drained)
  assert_eq(supervisor.survivor_phase, 1)

  terminal_finish_write(supervisor, 1)
  let closed = terminal_response(supervisor, 1, Closed)
  supervisor.accept_survivor_response(1, closed)
  assert_eq(supervisor.protocol.lifecycle(), Closed)
  assert_eq(supervisor.lifecycle(), Cleaning)
  let report = supervisor.take_group_failure()
  assert_eq(report.failed_rank(), 0)
  assert_eq(report.reason(), RankLost)
  terminal_reap(supervisor)
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "cleanup may close protocol before retained failure publication" {
  let PreparedRankGroupProcess(supervisor) = ownership_prepared()
  supervisor.protocol.report_idle_rank_loss(1)
  supervisor.record_failure(TransportLost(rank=1))
  supervisor.cleanup_lost[0] = true
  supervisor.protocol.mark_drain_rank_lost(0)
  supervisor.begin_cleanup()
  assert_eq(supervisor.protocol.lifecycle(), Closed)
  assert_eq(supervisor.lifecycle(), Cleaning)
  let report = supervisor.take_group_failure()
  assert_eq(report.failed_rank(), 1)
  assert_eq(report.reason(), RankLost)
  terminal_reap(supervisor)
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "survivor cleanup setup resumes after a fallible staging boundary" {
  let PreparedRankGroupProcess(supervisor) = ownership_prepared()
  terminal_live_children(supervisor)
  ownership_prime_wire_ready(supervisor)
  supervisor.lifecycle_value = Ready
  supervisor.protocol.report_idle_rank_loss(0)
  supervisor.record_failure(TransportLost(rank=0))
  // Model a setup attempt that persisted its exact phase but failed before an
  // in-flight survivor write was published. The retry must resume, not reject
  // the terminal guard as a duplicate cleanup owner.
  supervisor.survivor_terminal = true
  supervisor.survivor_phase = 0
  supervisor.survivor_rank = -1
  supervisor.io_rank = -1
  supervisor.io_phase = 0
  supervisor.begin_cleanup()
  assert_eq(supervisor.survivor_rank, 1)
  assert_eq(supervisor.io_phase, 1)

  terminal_finish_write(supervisor, 1)
  let drained = terminal_response(supervisor, 1, Drained)
  supervisor.accept_survivor_response(1, drained)
  terminal_finish_write(supervisor, 1)
  let closed = terminal_response(supervisor, 1, Closed)
  supervisor.accept_survivor_response(1, closed)
  assert_eq(supervisor.lifecycle(), Cleaning)
  terminal_reap(supervisor)
}
