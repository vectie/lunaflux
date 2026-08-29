///|
fn expect_root_binding_failure(
  source : @worker_wire.EncodedBootstrapSource,
  model : @approved_fs.ApprovedRoot,
  kernel : @approved_fs.ApprovedRoot,
  expected : RootBoundWorkerProcessFailure,
) -> Unit raise {
  try
    prepare_with_approved_roots(
      @worker_executable_fixture.cat_path(),
      supervisor_startup_for_source(source),
      source,
      supervisor_process_limits(),
      model,
      kernel,
    )
  catch {
    RootBoundFailure(actual) => assert_eq(actual, expected)
    _ => fail("unexpected root binding failure")
  } noraise {
    _ => fail("invalid root binding reached child spawn")
  }
}

///|
test "root-bound admission sanitizes model and kernel identity failures by role" {
  let model = @approved_fs.ApprovedRoot::open_absolute("/usr")
  let kernel = @approved_fs.ApprovedRoot::open_absolute("/usr")
  expect_root_binding_failure(
    supervisor_source_for_roots("/usr/bin", "/usr"),
    model,
    kernel,
    ModelRootBinding(Invalid(RootIdentity)),
  )
  expect_root_binding_failure(
    supervisor_source_for_roots("/usr", "/usr/bin"),
    model,
    kernel,
    KernelRootBinding(Invalid(RootIdentity)),
  )
  model.close()
  kernel.close()
}

///|
test "closed root binding failure is sanitized before retained pair acquisition" {
  let model = @approved_fs.ApprovedRoot::open_absolute("/usr")
  let kernel = @approved_fs.ApprovedRoot::open_absolute("/usr")
  model.close()
  let source = supervisor_source_for_roots("/usr", "/usr")
  expect_root_binding_failure(
    source,
    model,
    kernel,
    ModelRootBinding(Invalid(Lifecycle)),
  )
  assert_false(kernel.is_closed())
  kernel.close()
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "same approved root may satisfy both bootstrap roles" {
  let root = @approved_fs.ApprovedRoot::open_absolute("/usr")
  let source = supervisor_source_for_roots("/usr", "/usr")
  match
    prepare_with_approved_roots(
      @worker_executable_fixture.cat_path(),
      supervisor_startup_for_source(source),
      source,
      supervisor_process_limits(),
      root,
      root,
    ) {
    RootBoundCleanupRequired(failed) => {
      assert_true(failed.is_closed())
      assert_eq(failed.cause(), Process(Invalid(Submission)))
    }
    ReadyRootBoundProcess(_) =>
      fail("malformed child published same-root supervisor")
  }
  assert_false(root.is_closed())
  root.close()
}

///|
async test "namespace replacement cannot substitute a root-bound source" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-root-binding")
    let parent = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(parent, recursive=true))
    }
    let model_path = "\{parent}/model"
    let moved_path = "\{parent}/model.moved"
    let kernel_path = "\{parent}/kernel"
    @fs.mkdir(model_path)
    @fs.mkdir(kernel_path)
    let model = @approved_fs.ApprovedRoot::open_absolute(model_path)
    let kernel = @approved_fs.ApprovedRoot::open_absolute(kernel_path)
    @fs.rename(model_path, moved_path)
    @fs.mkdir(model_path)
    let source = supervisor_source_for_roots(model_path, kernel_path)
    expect_root_binding_failure(
      source,
      model,
      kernel,
      ModelRootBinding(Invalid(RootIdentity)),
    )
    assert_false(model.is_closed())
    assert_false(kernel.is_closed())
    model.close()
    kernel.close()
  }
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
async test "root-bound model and kernel roles authenticate an ordered pair" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-root-binding-order")
    let parent = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(parent, recursive=true))
    }
    let model_path = "\{parent}/model"
    let kernel_path = "\{parent}/kernel"
    @fs.mkdir(model_path)
    @fs.mkdir(kernel_path)
    let model = @approved_fs.ApprovedRoot::open_absolute(model_path)
    let kernel = @approved_fs.ApprovedRoot::open_absolute(kernel_path)
    let source = supervisor_source_for_roots(model_path, kernel_path)
    match
      prepare_with_approved_roots(
        @worker_executable_fixture.cat_path(),
        supervisor_startup_for_source(source),
        source,
        supervisor_process_limits(),
        model,
        kernel,
      ) {
      RootBoundCleanupRequired(failed) => {
        assert_true(failed.is_closed())
        assert_eq(failed.cause(), Process(Invalid(Submission)))
      }
      ReadyRootBoundProcess(_) =>
        fail("malformed child published ordered root pair")
    }
    expect_root_binding_failure(
      source,
      kernel,
      model,
      ModelRootBinding(Invalid(RootIdentity)),
    )
    model.close()
    kernel.close()
  }
}

///|
async test "root-bound identity follows no role label symlink" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-root-binding-link")
    let parent = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(parent, recursive=true))
    }
    let model_path = "\{parent}/model"
    let model_link = "\{parent}/model.link"
    let kernel_path = "\{parent}/kernel"
    @fs.mkdir(model_path)
    @fs.mkdir(kernel_path)
    @fs.symlink(target=model_path, model_link)
    let model = @approved_fs.ApprovedRoot::open_absolute(model_path)
    let kernel = @approved_fs.ApprovedRoot::open_absolute(kernel_path)
    let source = supervisor_source_for_roots(model_link, kernel_path)
    expect_root_binding_failure(
      source,
      model,
      kernel,
      ModelRootBinding(UnsupportedType(Root)),
    )
    model.close()
    kernel.close()
  }
}
