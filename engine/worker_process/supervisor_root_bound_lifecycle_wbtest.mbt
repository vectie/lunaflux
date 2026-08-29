///|
test "root-bound source mismatch precedes root acquisition" {
  let model = @approved_fs.ApprovedRoot::open_absolute("/usr")
  let kernel = @approved_fs.ApprovedRoot::open_absolute("/usr")
  model.close()
  kernel.close()
  try
    prepare_with_approved_roots(
      @worker_executable_fixture.cat_path(),
      supervisor_startup(),
      alternate_supervisor_source(),
      supervisor_process_limits(),
      model,
      kernel,
    )
  catch {
    RootBoundFailure(Process(Invalid(SourceMismatch))) => ()
    _ => fail("source mismatch acquired closed roots")
  } noraise {
    _ => fail("mismatched source reached root acquisition")
  }
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "root-bound restart retains source and has no substitution input" {
  let owner = test_root_bound_supervisor()
  let digest = owner.bootstrap_source().digest()
  owner.begin_recovery()
  assert_eq(owner.lifecycle(), RootBoundRecovering)
  try owner.restart() catch {
    RootBoundFailure(Process(Invalid(Submission))) => ()
    _ => fail("unexpected rooted replacement failure")
  } noraise {
    _ => fail("cat fixture unexpectedly completed startup")
  }
  assert_eq(owner.lifecycle(), RootBoundRecovering)
  assert_eq(owner.bootstrap_source().digest(), digest)
  owner.close()
  owner.close()
  assert_true(owner.is_closed())
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "delegated fatal transport failure marks recovery required" {
  let owner = test_root_bound_supervisor()
  let plan = supervisor_plan(1UL)
  let submitted = owner.submit(plan)
  try owner.receive(submitted, plan) catch {
    RootBoundFailure(Process(Transport(InvalidConfiguration))) => ()
    _ => fail("unexpected wrapped malformed response")
  } noraise {
    _ => fail("echoed plan became a completion")
  }
  assert_eq(owner.lifecycle(), RootBoundRecoveryRequired)
  owner.close()
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "recovery transition consumes only child and retains roots for restart" {
  let owner = test_root_bound_supervisor()
  owner.begin_recovery()
  assert_eq(owner.lifecycle(), RootBoundRecovering)
  assert_eq(owner.child_cleanup_failure(), None)
  assert_false(owner.is_closed())
  owner.close()
  assert_eq(owner.lifecycle(), RootBoundClosed)
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "root-bound recovery validates scalars without exporting startup" {
  let owner = test_root_bound_supervisor()
  let startup = supervisor_startup()
  owner.begin_recovery()
  owner.validate_recovery_binding(
    startup.identity(),
    startup.bootstrap_digest(),
    startup.bootstrap_source_digest(),
    startup.device_ordinal(),
    startup.model_generation_value(),
    startup.worker_limits(),
    startup.inference_limits(),
    0UL,
  )
  try
    owner.validate_recovery_binding(
      startup.identity(),
      startup.bootstrap_digest(),
      startup.bootstrap_source_digest(),
      startup.device_ordinal(),
      startup.model_generation_value() + 1UL,
      startup.worker_limits(),
      startup.inference_limits(),
      0UL,
    )
  catch {
    RootBoundOwnerMismatch => ()
    _ => fail("unexpected recovery binding rejection")
  } noraise {
    _ => fail("foreign recovery generation was admitted")
  }
  owner.close()
}
