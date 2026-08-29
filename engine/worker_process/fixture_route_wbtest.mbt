///|
fn[T] close_fixture_and_raise_root(
  admitted : @worker_executable_file.WorkerExecutableAdmission,
  error : RootBoundWorkerProcessError,
) -> T raise RootBoundWorkerProcessError {
  try admitted.close() catch {
    _ => raise error
  } noraise {
    _ => raise error
  }
}

///|
fn[T] close_fixture_and_raise(
  admitted : @worker_executable_file.WorkerExecutableAdmission,
  error : WorkerProcessError,
) -> T raise WorkerProcessError {
  try admitted.close() catch {
    _ => raise error
  } noraise {
    _ => raise error
  }
}

///|
fn prepare_with_approved_roots(
  executable : Bytes,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  limits : WorkerProcessLimits,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
) -> RootBoundWorkerProcessPreparation raise RootBoundWorkerProcessError {
  let admitted = @worker_executable_fixture.admit(executable) catch {
    _ => raise RootBoundFailure(Process(Invalid(Submission)))
  }
  let result = try
    prepare_with_approved_executable(
      admitted, startup, source, limits, model_root, kernel_root,
    )
  catch {
    error => close_fixture_and_raise_root(admitted, error)
  } noraise {
    value => value
  }
  match result {
    RootBoundCleanupRequired(failed) => {
      failed.retry_close()
      result
    }
    ReadyRootBoundProcess(_) => result
  }
}

///|
fn prepare_exchange_with_approved_roots(
  executable : Bytes,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  limits : WorkerProcessLimits,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  expected_predecessor : UInt64,
) -> RootBoundWorkerProcessPreparation raise RootBoundWorkerProcessError {
  let admitted = @worker_executable_fixture.admit(executable) catch {
    _ => raise RootBoundFailure(Process(Invalid(Submission)))
  }
  let result = try
    prepare_exchange_with_approved_executable(
      admitted, startup, source, limits, model_root, kernel_root, expected_predecessor,
    )
  catch {
    error => close_fixture_and_raise_root(admitted, error)
  } noraise {
    value => value
  }
  match result {
    RootBoundCleanupRequired(failed) => {
      failed.retry_close()
      result
    }
    ReadyRootBoundProcess(_) => result
  }
}

///|
fn spawn_worker_fixture_child(
  path : Bytes,
  limits : @process.ProcessLimits,
) -> @process.ChildProcess raise {
  let executable = @worker_executable_fixture.admit(path)
  let model = @approved_fs.ApprovedRoot::open_absolute("/usr")
  let kernel = @approved_fs.ApprovedRoot::open_absolute("/usr")
  let roots = @approved_fs.acquire_worker_approved_roots(
    model_root=model,
    kernel_root=kernel,
  )
  let child = executable
    .prepare_child(limits)
    .spawn_with_approved_roots(
      @approved_fs.worker_approved_root_spawn_authority(roots),
    )
  @approved_fs.close_worker_approved_roots(roots)
  model.close()
  kernel.close()
  executable.close()
  child
}

///|
/// Test-only non-rooted protocol fixture. Linux executes through the production
/// pinned-descriptor route; unsupported hosts fail closed at spawn.
fn prepare(
  path : Bytes,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  limits : WorkerProcessLimits,
) -> WorkerProcessPreparation raise {
  let executable = @worker_executable_fixture.admit(path) catch {
    _ => raise Invalid(Submission)
  }
  let prepared = try
    preflight_prepare_approved(executable, startup, source, limits, None)
  catch {
    error => close_fixture_and_raise(executable, error)
  } noraise {
    value => value
  }
  let model = @approved_fs.ApprovedRoot::open_absolute("/usr")
  let kernel = @approved_fs.ApprovedRoot::open_absolute("/usr")
  let roots = @approved_fs.acquire_worker_approved_roots(
    model_root=model,
    kernel_root=kernel,
  )
  let result = prepare_preflighted(source, limits, prepared, roots)
  @approved_fs.close_worker_approved_roots(roots)
  model.close()
  kernel.close()
  executable.close()
  match result {
    CleanupRequired(failed) if failed.is_closed() => raise failed.cause()
    _ => result
  }
}
