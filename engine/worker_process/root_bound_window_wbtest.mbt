///|

///|
fn test_root_bound_supervisor() -> RootBoundWorkerProcessSupervisor raise {
  let model = @approved_fs.ApprovedRoot::open_absolute("/usr")
  let kernel = @approved_fs.ApprovedRoot::open_absolute("/usr")
  let roots = @approved_fs.acquire_worker_approved_roots(
    model_root=model,
    kernel_root=kernel,
  )
  model.close()
  kernel.close()
  {
    approved_executable: @worker_executable_fixture.admit(
      @worker_executable_fixture.cat_path(),
    ),
    limits: supervisor_process_limits(),
    roots,
    roots_selected: true,
    process: test_supervisor(),
    failed_replacement: None,
    cause: None,
    child_cleanup_failure: None,
    roots_cleanup_failure: None,
    transport_mode: 0,
    exchange_epoch: 0UL,
    active_exchange_epoch: 0UL,
    exchange_phase: -1,
    exchange_side: -1,
    exchange_sequence: 0UL,
    exchange_plan: @worker_protocol.SubmittedSchedulePlanSlot::new(),
    maintenance_mode: 0,
    lifecycle: RootBoundReady,
  }
}

///|
test "closed executable rejection precedes closed-root acquisition" {
  let model = @approved_fs.ApprovedRoot::open_absolute("/usr")
  let kernel = @approved_fs.ApprovedRoot::open_absolute("/usr")
  let source = supervisor_source_for_roots("/usr", "/usr")
  let executable = @worker_executable_fixture.admit(
    @worker_executable_fixture.cat_path(),
  )
  executable.close()
  model.close()
  kernel.close()
  try
    prepare_with_approved_executable(
      executable,
      supervisor_startup_for_source(source),
      source,
      supervisor_process_limits(),
      model,
      kernel,
    )
  catch {
    RootBoundFailure(Process(Transport(SpawnFailed))) => ()
    _ => fail("executable preflight reached root acquisition")
  } noraise {
    _ => fail("closed executable passed rooted preflight")
  }
}

///|
test "rooted preparation rejects frame capacity before acquiring closed roots" {
  let model = @approved_fs.ApprovedRoot::open_absolute("/usr")
  let kernel = @approved_fs.ApprovedRoot::open_absolute("/usr")
  model.close()
  kernel.close()
  let limits = WorkerProcessLimits::new(
    max_frame_bytes=64,
    startup_io_timeout_millis=100,
    io_timeout_millis=100,
    shutdown_timeout_millis=100,
  )
  try
    prepare_with_approved_roots(
      @worker_executable_fixture.cat_path(),
      supervisor_startup(),
      supervisor_source(),
      limits,
      model,
      kernel,
    )
  catch {
    RootBoundFailure(Process(Invalid(Capacity))) => ()
    _ => fail("capacity preflight reached root acquisition")
  } noraise {
    _ => fail("undersized frame capacity passed rooted preflight")
  }
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "root-bound production supervisor serializes the device child window" {
  let owner = test_root_bound_supervisor()
  let first = owner.submit(supervisor_plan(1UL))
  assert_false(owner.has_capacity())
  try owner.submit(supervisor_plan(2UL)) catch {
    RootBoundFailure(Process(Backpressured)) => ()
    _ => fail("unexpected root-bound second-submit failure")
  } noraise {
    _ => fail("root-bound supervisor admitted a second in-flight plan")
  }
  assert_eq(first.sequence_value(), 1UL)
  owner.close()
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "legacy and nonblocking exchange modes cannot mix" {
  let legacy = test_root_bound_supervisor()
  ignore(legacy.submit(supervisor_plan(1UL)))
  try legacy.begin_exchange(supervisor_plan(2UL)) catch {
    RootBoundFailure(Process(Backpressured)) => ()
    _ => fail("unexpected legacy-to-exchange error")
  } noraise {
    _ => fail("exchange mixed with legacy submission")
  }
  legacy.close()
  let exchange_owner = test_root_bound_supervisor()
  let sequence = exchange_owner.begin_exchange(supervisor_plan(1UL))
  assert_eq(sequence, 1UL)
  try exchange_owner.submit(supervisor_plan(2UL)) catch {
    RootBoundFailure(Process(Invalid(Lifecycle))) => ()
    _ => fail("unexpected exchange-to-legacy error")
  } noraise {
    _ => fail("legacy submission mixed with exchange")
  }
  exchange_owner.close()
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "exchange claim rejects legacy and previously used exchange owners" {
  let legacy = test_root_bound_supervisor()
  ignore(legacy.submit(supervisor_plan(1UL)))
  try legacy.claim_exchange(0UL) catch {
    RootBoundFailure(Process(Invalid(Lifecycle))) => ()
    _ => fail("unexpected legacy claim rejection")
  } noraise {
    _ => fail("legacy-used owner admitted an exchange service claim")
  }
  legacy.close()

  let used = test_root_bound_supervisor()
  let plan = supervisor_plan(1UL, completion_slot=1)
  ignore(used.begin_exchange(plan))
  inject_exchange_completion(used, plan)
  let received = used.received_for_sequence(1UL)
  used.clear_exchange()
  used.retire_received(received)
  try used.claim_exchange(1UL) catch {
    RootBoundFailure(Process(Invalid(Lifecycle))) => ()
    _ => fail("unexpected reused exchange claim rejection")
  } noraise {
    _ => fail("previously used exchange owner was rebound to a service")
  }
  used.close()
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "malformed nonblocking response fail-stops the rooted supervisor" {
  let owner = test_root_bound_supervisor()
  let active_plan = supervisor_plan(1UL)
  ignore(owner.begin_exchange(active_plan))
  let mut attempts = 0
  let mut failed = false
  while attempts < 10000 && !failed {
    try owner.progress_exchange() catch {
      RootBoundFailure(Process(_)) => failed = true
      _ => fail("unexpected exchange progression failure")
    } noraise {
      _ => ()
    }
    attempts += 1
  }
  assert_true(failed)
  assert_eq(owner.lifecycle(), RootBoundRecoveryRequired)
  try owner.progress_exchange() catch {
    RootBoundInvalidLifecycle => ()
    _ => fail("unexpected stale exchange error")
  } noraise {
    _ => fail("failed exchange remained usable")
  }
  owner.close()
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "post-reservation begin failure retains exact process obligation" {
  let owner = test_root_bound_supervisor()
  owner.process.require_child().close()
  try owner.begin_exchange(supervisor_plan(1UL)) catch {
    RootBoundFailure(Process(Transport(InvalidState))) => ()
    _ => fail("unexpected post-reservation begin failure")
  } noraise {
    _ => fail("closed child began an exchange")
  }
  assert_eq(owner.lifecycle(), RootBoundRecoveryRequired)
  assert_eq(owner.submission_for_sequence(1UL).sequence_value(), 1UL)
  owner.close()
}
