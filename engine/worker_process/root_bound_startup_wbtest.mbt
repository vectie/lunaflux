///|
test "exchange predecessor mismatch precedes closed-root acquisition" {
  let model = @approved_fs.ApprovedRoot::open_absolute("/usr")
  let kernel = @approved_fs.ApprovedRoot::open_absolute("/usr")
  model.close()
  kernel.close()
  try
    prepare_exchange_with_approved_roots(
      @worker_executable_fixture.cat_path(),
      supervisor_startup(),
      supervisor_source(),
      supervisor_process_limits(),
      model,
      kernel,
      1UL,
    )
  catch {
    RootBoundFailure(Process(Invalid(Sequence))) => ()
    _ => fail("unexpected predecessor preflight failure")
  } noraise {
    _ => fail("predecessor mismatch reached root acquisition")
  }
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "rooted malformed ready closes child and leaves caller roots independent" {
  let model = @approved_fs.ApprovedRoot::open_absolute("/usr")
  let kernel = @approved_fs.ApprovedRoot::open_absolute("/usr")
  let source = supervisor_source_for_roots("/usr", "/usr")
  match
    prepare_exchange_with_approved_roots(
      @worker_executable_fixture.cat_path(),
      supervisor_startup_for_source(source),
      source,
      supervisor_process_limits(),
      model,
      kernel,
      0UL,
    ) {
    RootBoundCleanupRequired(failed) => {
      assert_true(failed.is_closed())
      assert_eq(failed.cause(), Process(Invalid(Submission)))
    }
    ReadyRootBoundProcess(_) => fail("cat response published rooted owner")
  }
  model.close()
  kernel.close()
  assert_true(model.is_closed())
  assert_true(kernel.is_closed())
}
