///|
fn close_executor(
  resources : DeviceWorkerResources,
) -> Unit raise DeviceWorkerError {
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
  resources.executor = None
}

///|
fn close_weights(
  resources : DeviceWorkerResources,
) -> Unit raise DeviceWorkerError {
  match resources.weights {
    Some(Complete(owner)) =>
      owner.close() catch {
        error => raise CleanupWeights(error)
      }
    Some(Failed(owner)) =>
      owner.retry_close() catch {
        error => raise CleanupWeights(error)
      }
    None => return
  }
  resources.weights = None
}

///|
fn close_context(
  resources : DeviceWorkerResources,
) -> Unit raise DeviceWorkerError {
  match resources.context {
    Some(owner) => owner.close() catch { error => raise CleanupContext(error) }
    None => return
  }
  resources.context = None
}

///|
fn resources_complete(resources : DeviceWorkerResources) -> Bool {
  resources.executor is Some(Complete(_)) &&
  resources.weights is Some(Complete(_)) &&
  resources.context is Some(_)
}

///|
fn close_resources(
  resources : DeviceWorkerResources,
) -> Unit raise DeviceWorkerError {
  run_cleanup_chain(
    () => close_executor(resources),
    () => close_weights(resources),
    () => close_context(resources),
  )
}

///|
fn run_cleanup_chain(
  executor : () -> Unit raise DeviceWorkerError,
  weights : () -> Unit raise DeviceWorkerError,
  context : () -> Unit raise DeviceWorkerError,
) -> Unit raise DeviceWorkerError {
  // Dependency gates are deliberate: a parent is never closed while a child
  // still retains its lifetime.
  executor()
  weights()
  context()
}

///|
fn preparation_failed(
  resources : DeviceWorkerResources,
  cause : DeviceWorkerError,
) -> DeviceWorkerPreparation raise DeviceWorkerError {
  try close_resources(resources) catch {
    cleanup_failure =>
      DeviceWorkerCleanupRequired({
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
/// Closes executor, weights, then context. A failure retains the owner and
/// blocks release of every parent dependency; repeated calls retry safely.
pub fn DeviceWorkerOwner::close(
  self : DeviceWorkerOwner,
) -> Unit raise DeviceWorkerError {
  if self.lifecycle == WorkerClosed {
    return
  }
  self.lifecycle = WorkerClosingFailed
  close_resources(self.resources)
  self.lifecycle = WorkerClosed
}

///|
/// Retries dependency-ordered cleanup. Success and later calls are idempotent;
/// failure preserves the exact remaining authority.
pub fn FailedDeviceWorkerPreparation::retry_close(
  self : FailedDeviceWorkerPreparation,
) -> Unit raise DeviceWorkerError {
  if self.lifecycle == WorkerClosed {
    return
  }
  self.lifecycle = WorkerClosingFailed
  close_resources(self.resources) catch {
    error => {
      self.cleanup_failure = error
      raise error
    }
  }
  self.lifecycle = WorkerClosed
}
