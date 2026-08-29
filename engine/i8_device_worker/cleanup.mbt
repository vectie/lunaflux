///|
fn close_executor(
  resources : I8DeviceWorkerResources,
) -> Unit raise I8DeviceWorkerError {
  let expected = resources.owner_phase
  match resources.executor {
    Some(_) =>
      match expected {
        ReadyExecutorOwner | FailedExecutorOwner => ()
        _ => raise Invalid(ResourceState)
      }
    None => return
  }
  match resources.executor {
    Some(Complete(owner)) =>
      owner.close() catch {
        error => raise CleanupExecutor(error)
      }
    Some(Failed(owner)) =>
      owner.retry_close() catch {
        error => raise CleanupExecutor(error)
      }
    None => return
  }
  commit_owner_transition(resources, expected, OwnerSlotsEmpty, () => {
    resources.executor = None
  })
}

///|
fn close_numeric(
  resources : I8DeviceWorkerResources,
) -> Unit raise I8DeviceWorkerError {
  let expected = resources.owner_phase
  match resources.numeric {
    Some(_) =>
      match expected {
        RawNumericOwner
        | AuthenticatedNumericOwner
        | LoadedNumericOwner
        | ExecutorTransferOwner
        | FailedNumericOwner => ()
        _ => raise Invalid(ResourceState)
      }
    None => return
  }
  match resources.numeric {
    Some(Raw(owner)) =>
      owner.close() catch {
        error => raise CleanupNumericWeights(error)
      }
    Some(Authenticated(owner)) =>
      owner.close() catch {
        error => raise CleanupNumericWeights(error)
      }
    Some(Loaded(owner)) =>
      owner.close() catch {
        error => raise CleanupExecutor(error)
      }
    Some(ExecutorTransfer(owner)) =>
      owner.close() catch {
        error => raise CleanupExecutionManifest(error)
      }
    Some(Failed(owner)) =>
      owner.retry_close() catch {
        error => raise CleanupNumericWeights(error)
      }
    None => return
  }
  commit_owner_transition(resources, expected, OwnerSlotsEmpty, () => {
    resources.numeric = None
  })
}

///|
fn close_context(
  resources : I8DeviceWorkerResources,
) -> Unit raise I8DeviceWorkerError {
  match resources.context {
    Some(owner) => owner.close() catch { error => raise CleanupContext(error) }
    None => return
  }
  resources.context = None
}

///|
fn resources_complete(resources : I8DeviceWorkerResources) -> Bool {
  i8_resources_publishable(
    WorkerReady,
    resources.executor is Some(Complete(_)),
    resources.numeric is Some(_),
    resources.context is Some(_),
    resources.owner_phase == ReadyExecutorOwner,
  )
}

///|
fn i8_resources_publishable(
  lifecycle : I8DeviceWorkerLifecycle,
  executor_complete : Bool,
  numeric_pending : Bool,
  context_present : Bool,
  exact_owner_phase : Bool,
) -> Bool {
  lifecycle == WorkerReady &&
  executor_complete &&
  !numeric_pending &&
  context_present &&
  exact_owner_phase
}

///|
fn run_i8_cleanup_chain(
  executor : () -> Unit raise I8DeviceWorkerError,
  numeric : () -> Unit raise I8DeviceWorkerError,
  context : () -> Unit raise I8DeviceWorkerError,
) -> Unit raise I8DeviceWorkerError {
  executor()
  numeric()
  context()
}

///|
fn close_resources(
  resources : I8DeviceWorkerResources,
) -> Unit raise I8DeviceWorkerError {
  // Dependency gates are deliberate. A parent is never closed while an exact
  // child owner still retains its lifetime.
  run_i8_cleanup_chain(
    () => close_executor(resources),
    () => close_numeric(resources),
    () => close_context(resources),
  )
}

///|
fn preparation_failed(
  resources : I8DeviceWorkerResources,
  cause : I8DeviceWorkerError,
) -> I8DeviceWorkerPreparation raise I8DeviceWorkerError {
  try close_resources(resources) catch {
    cleanup_failure =>
      I8DeviceWorkerCleanupRequired({
        resources,
        cause,
        cleanup_failure,
        lifecycle: WorkerClosingFailed,
      })
  } noraise {
    _ => raise cause
  }
}

///|
pub fn I8DeviceWorkerOwner::close(
  self : I8DeviceWorkerOwner,
) -> Unit raise I8DeviceWorkerError {
  if self.lifecycle == WorkerClosed {
    return
  }
  self.lifecycle = WorkerClosingFailed
  close_resources(self.resources)
  self.lifecycle = WorkerClosed
}

///|
pub fn FailedI8DeviceWorkerPreparation::retry_close(
  self : FailedI8DeviceWorkerPreparation,
) -> Unit raise I8DeviceWorkerError {
  if self.lifecycle == WorkerClosed {
    return
  }
  self.lifecycle = WorkerClosingFailed
  let outcome : Result[Unit, I8DeviceWorkerError] = Ok(
    close_resources(self.resources),
  ) catch {
    error => Err(error)
  }
  match outcome {
    Err(error) => {
      self.cleanup_failure = error
      raise error
    }
    Ok(_) => ()
  }
  self.lifecycle = WorkerClosed
}
