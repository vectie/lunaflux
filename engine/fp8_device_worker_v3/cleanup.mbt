///|
fn close_resources_v3(
  resources : Fp8DeviceWorkerV3Resources,
) -> Unit raise Fp8DeviceWorkerV3Error {
  match resources.executor {
    Some(Complete(owner)) =>
      owner.close() catch {
        error => raise CleanupExecutor(error)
      }
    Some(Failed(owner)) =>
      owner.retry_close() catch {
        error => raise CleanupExecutor(error)
      }
    None => ()
  }
  resources.executor = None
  match resources.numeric {
    Some(Raw(owner)) =>
      owner.close() catch {
        error => raise CleanupNumericWeights(error)
      }
    Some(Authenticated(owner)) =>
      owner.close() catch {
        error => raise CleanupNumericWeights(error)
      }
    Some(Failed(owner)) =>
      owner.retry_close() catch {
        error => raise CleanupNumericWeights(error)
      }
    None => ()
  }
  resources.numeric = None
  match resources.context {
    Some(owner) => owner.close() catch { error => raise CleanupContext(error) }
    None => ()
  }
  resources.context = None
}

///|
fn remember_worker_cleanup_failure_v3(
  failed : FailedFp8DeviceWorkerPreparationV3,
  error : Fp8DeviceWorkerV3Error,
) -> Unit raise Fp8DeviceWorkerV3Error {
  failed.cleanup_failure = error
  raise error
}

///|
pub fn Fp8DeviceWorkerOwnerV3::close(
  self : Self,
) -> Unit raise Fp8DeviceWorkerV3Error {
  if self.lifecycle == WorkerClosed {
    return
  }
  self.lifecycle = WorkerClosingFailed
  close_resources_v3(self.resources)
  self.lifecycle = WorkerClosed
}

///|
pub fn FailedFp8DeviceWorkerPreparationV3::retry_close(
  self : Self,
) -> Unit raise Fp8DeviceWorkerV3Error {
  if self.lifecycle == WorkerClosed {
    return
  }
  self.lifecycle = WorkerClosingFailed
  close_resources_v3(self.resources) catch {
    error => remember_worker_cleanup_failure_v3(self, error)
  }
  self.lifecycle = WorkerClosed
}
