///|
fn close_rank_child_root(
  root : @approved_fs.ApprovedRoot,
) -> Unit raise TensorParallelRankChildError {
  root.close() catch {
    error => raise Root(error)
  }
}

///|
fn close_rank_child_roots_after_failure(
  model_root : @approved_fs.ApprovedRoot?,
  kernel_root : @approved_fs.ApprovedRoot?,
) -> Unit {
  match kernel_root {
    Some(root) => root.close() catch { _ => () }
    None => ()
  }
  match model_root {
    Some(root) => root.close() catch { _ => () }
    None => ()
  }
}

///|
fn open_rank_child_roots(
  configured : TensorParallelRankChildConfigured,
) -> (@approved_fs.ApprovedRoot, @approved_fs.ApprovedRoot) raise TensorParallelRankChildError {
  let roots = configured.contract.bootstrap_source().roots()
  let model = @approved_fs.ApprovedRoot::open_inherited(Model) catch {
    error => raise Root(error)
  }
  model.require_absolute_identity(roots.model_root()) catch {
    error => {
      close_rank_child_roots_after_failure(Some(model), None)
      raise Root(error)
    }
  }
  let kernel = @approved_fs.ApprovedRoot::open_inherited(Kernel) catch {
    error => {
      close_rank_child_roots_after_failure(Some(model), None)
      raise Root(error)
    }
  }
  kernel.require_absolute_identity(roots.kernel_root()) catch {
    error => {
      close_rank_child_roots_after_failure(Some(model), Some(kernel))
      raise Root(error)
    }
  }
  (model, kernel)
}

///|
fn failed_rank_child_worker(
  worker : TensorParallelRankChildCleanupAuthority,
  cause : TensorParallelRankChildError,
  cleanup_failure : TensorParallelRankChildError?,
) -> TensorParallelRankChildPreparation {
  RankChildCleanupRequired({ worker, cause, cleanup_failure, closed: false })
}

///|
pub fn prepare_configured_rank_child(
  configured : TensorParallelRankChildConfigured,
) -> TensorParallelRankChildPreparation raise TensorParallelRankChildError {
  let roots = open_rank_child_roots(configured)
  let mut admission_complete = false
  defer (if !admission_complete {
    close_rank_child_roots_after_failure(Some(roots.0), Some(roots.1))
  })
  let admitted = build_rank_child_admission(configured, roots.0, roots.1)
  admission_complete = true
  let mut kernel_closed = false
  defer (if !kernel_closed {
    close_rank_child_roots_after_failure(Some(admitted.model_root), None)
  })
  close_rank_child_root(admitted.kernel_root)
  kernel_closed = true
  let prepared = @tensor_parallel_device_worker.prepare(
    admitted.model_root,
    admitted.collective_runtime,
    admitted.worker_plan,
  ) catch {
    error => {
      close_rank_child_roots_after_failure(Some(admitted.model_root), None)
      raise Worker(error)
    }
  }
  match prepared {
    TensorParallelWorkerPrepared(worker) => {
      try close_rank_child_root(admitted.model_root) catch {
        root_failure =>
          try worker.close() catch {
            cleanup_failure =>
              return failed_rank_child_worker(
                CleanupLiveWorker(worker),
                root_failure,
                Some(Worker(cleanup_failure)),
              )
          } noraise {
            _ => raise root_failure
          }
      } noraise {
        _ => ()
      }
      try begin_worker_startup(configured, worker) catch {
        startup_failure =>
          try worker.close() catch {
            cleanup_failure =>
              return failed_rank_child_worker(
                CleanupLiveWorker(worker),
                startup_failure,
                Some(Worker(cleanup_failure)),
              )
          } noraise {
            _ => raise startup_failure
          }
      } noraise {
        starting => RankChildStarting(starting)
      }
    }
    TensorParallelWorkerCleanupRequired(failed) => {
      let root_failure : TensorParallelRankChildError? = try
        close_rank_child_root(admitted.model_root)
      catch {
        error => Some(error)
      } noraise {
        _ => None
      }
      let primary = Worker(failed.cause())
      let cleanup = match root_failure {
        Some(error) => Some(error)
        None => Some(Worker(failed.cleanup_failure()))
      }
      failed_rank_child_worker(CleanupFailedWorker(failed), primary, cleanup)
    }
  }
}

///|
pub fn FailedTensorParallelRankChildPreparation::cause(
  self : FailedTensorParallelRankChildPreparation,
) -> TensorParallelRankChildError {
  self.cause
}

///|
pub fn FailedTensorParallelRankChildPreparation::cleanup_failure(
  self : FailedTensorParallelRankChildPreparation,
) -> TensorParallelRankChildError? {
  self.cleanup_failure
}

///|
pub fn FailedTensorParallelRankChildPreparation::is_closed(
  self : FailedTensorParallelRankChildPreparation,
) -> Bool {
  self.closed
}

///|
pub fn FailedTensorParallelRankChildPreparation::retry_close(
  self : FailedTensorParallelRankChildPreparation,
) -> Unit raise TensorParallelRankChildError {
  if self.closed {
    return
  }
  match self.worker {
    CleanupLiveWorker(worker) =>
      worker.close() catch {
        error => {
          self.cleanup_failure = Some(Worker(error))
          raise Worker(error)
        }
      }
    CleanupFailedWorker(worker) =>
      worker.retry_close() catch {
        error => {
          self.cleanup_failure = Some(Worker(error))
          raise Worker(error)
        }
      }
  }
  self.cleanup_failure = None
  self.closed = true
}
