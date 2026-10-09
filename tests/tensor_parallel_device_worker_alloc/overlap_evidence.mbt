///|
extern "C" fn verify_overlap_evidence() = "lunaflux_tp_alloc_verify_overlap"

///|
extern "C" fn fail_event_close_once() = "lunaflux_tp_alloc_fail_event_close_once"

///|
fn close_overlap_workers(fixture : WorkerFixture) -> Unit raise {
  for worker in fixture.workers {
    assert_true(worker.parallel_policy() == ComputeCommunicationOverlap)
    worker.begin_drain()
    fail_event_close_once()
    try worker.close() catch {
      @tensor_parallel_device_worker.Device(ParallelEvent, _) => ()
      _ => fail("unexpected cleanup failure")
    } noraise {
      _ => fail("event close failure was lost")
    }
    assert_true(worker.lifecycle() == WorkerClosingFailed)
    worker.close()
    assert_true(worker.lifecycle() == WorkerClosed)
  }
}

///|
fn abort_overlap_workers(
  fixture : WorkerFixture,
  cycle : PreparedCycle,
) -> Unit raise {
  for worker in fixture.workers {
    let staged = worker.stage_frame(cycle.frame)
    // Issue compute and one collective, then leave its GPU work pending.
    for _ in 0..<3 {
      assert_true(worker.progress_execution(staged, cycle.frame).is_pending())
    }
    worker.abort_plan()
    worker.close()
    assert_true(worker.lifecycle() == WorkerClosed)
  }
}
