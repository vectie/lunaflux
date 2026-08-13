///|
test "exchange predecessor mismatch precedes closed-root acquisition" {
  let model = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let kernel = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  model.close()
  kernel.close()
  try
    prepare_exchange_with_approved_roots(
      b"/bin/cat",
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
test "rooted malformed ready closes child and leaves caller roots independent" {
  let model = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let kernel = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  try
    prepare_exchange_with_approved_roots(
      b"/bin/cat",
      supervisor_startup(),
      supervisor_source(),
      supervisor_process_limits(),
      model,
      kernel,
      0UL,
    )
  catch {
    RootBoundFailure(Process(Invalid(Submission))) => ()
    _ => fail("unexpected malformed ready failure")
  } noraise {
    _ => fail("cat response published rooted owner")
  }
  model.close()
  kernel.close()
  assert_true(model.is_closed())
  assert_true(kernel.is_closed())
}
