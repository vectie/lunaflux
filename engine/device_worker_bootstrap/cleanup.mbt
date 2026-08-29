///|
fn root_error(
  error : @approved_fs.ApprovedFsError,
  role : BootstrapRootRole,
) -> DeviceWorkerBootstrapError {
  let issue = match error {
    Unavailable(_) => Unavailable
    UnsupportedType(_) => UnsupportedType
    Invalid(_) => AuthorityClosed
    Busy(_) => Busy
    Truncated | ChangedDuringRead | FileTooLarge | OperationFailed(_) => Failed
  }
  Root(role~, issue~)
}

///|
fn cleanup_root_error(
  error : @approved_fs.ApprovedFsError,
  role : BootstrapRootRole,
) -> DeviceWorkerBootstrapError {
  let issue = match root_error(error, role) {
    Root(issue~, ..) => issue
    _ => Failed
  }
  CleanupRoot(role~, issue~)
}

///|
fn close_model_root(
  roots : BootstrapRoots,
) -> Unit raise DeviceWorkerBootstrapError {
  match roots.model {
    Some(root) => {
      root.close() catch {
        error => raise cleanup_root_error(error, ModelRoot)
      }
      roots.model = None
    }
    None => ()
  }
}

///|
fn close_kernel_root(
  roots : BootstrapRoots,
) -> Unit raise DeviceWorkerBootstrapError {
  match roots.kernel {
    Some(root) => {
      root.close() catch {
        error => raise cleanup_root_error(error, KernelRoot)
      }
      roots.kernel = None
    }
    None => ()
  }
}

///|
fn close_worker_authority(
  authority : BootstrapWorkerAuthority,
) -> Unit raise DeviceWorkerBootstrapError {
  match authority {
    LegacyReady(owner) =>
      owner.close() catch {
        error => raise CleanupWorker(error)
      }
    LegacyFailed(owner) =>
      owner.retry_close() catch {
        error => raise CleanupWorker(error)
      }
    I8Ready(owner) =>
      owner.close() catch {
        error => raise CleanupI8Worker(error)
      }
    I8Failed(owner) =>
      owner.retry_close() catch {
        error => raise CleanupI8Worker(error)
      }
    Fp8Ready(owner) =>
      owner.close() catch {
        error => raise CleanupFp8Worker(error)
      }
    Fp8Failed(owner) =>
      owner.retry_close() catch {
        error => raise CleanupFp8Worker(error)
      }
  }
}

///|
fn close_worker_slot(
  slot : Ref[BootstrapWorkerAuthority?],
) -> Unit raise DeviceWorkerBootstrapError {
  match slot.val {
    Some(authority) => {
      close_worker_authority(authority)
      slot.val = None
    }
    None => ()
  }
}

///|
fn cleanup_all(
  roots : BootstrapRoots,
  worker : Ref[BootstrapWorkerAuthority?],
) -> (
  DeviceWorkerBootstrapError?,
  DeviceWorkerBootstrapError?,
  DeviceWorkerBootstrapError?,
) {
  cleanup_all_with(
    () => close_worker_slot(worker),
    () => close_model_root(roots),
    () => close_kernel_root(roots),
  )
}

///|
fn cleanup_all_with(
  close_worker : () -> Unit raise DeviceWorkerBootstrapError,
  close_model : () -> Unit raise DeviceWorkerBootstrapError,
  close_kernel : () -> Unit raise DeviceWorkerBootstrapError,
) -> (
  DeviceWorkerBootstrapError?,
  DeviceWorkerBootstrapError?,
  DeviceWorkerBootstrapError?,
) {
  let worker_failure : Ref[DeviceWorkerBootstrapError?] = Ref(None)
  let model_failure : Ref[DeviceWorkerBootstrapError?] = Ref(None)
  let kernel_failure : Ref[DeviceWorkerBootstrapError?] = Ref(None)
  close_worker() catch {
    error => worker_failure.val = Some(error)
  }
  close_model() catch {
    error => model_failure.val = Some(error)
  }
  close_kernel() catch {
    error => kernel_failure.val = Some(error)
  }
  (worker_failure.val, model_failure.val, kernel_failure.val)
}

///|
fn preparation_failed(
  roots : BootstrapRoots,
  worker : BootstrapWorkerAuthority?,
  cause : DeviceWorkerBootstrapError,
) -> DeviceWorkerBootstrapPreparation raise DeviceWorkerBootstrapError {
  let slot = Ref(worker)
  let (worker_failure, model_failure, kernel_failure) = cleanup_all(roots, slot)
  if worker_failure is None && model_failure is None && kernel_failure is None {
    raise cause
  }
  CleanupRequired({
    roots,
    worker: slot.val,
    cause,
    worker_cleanup_failure: worker_failure,
    model_cleanup_failure: model_failure,
    kernel_cleanup_failure: kernel_failure,
    closed: false,
  })
}

///|
pub fn FailedDeviceWorkerBootstrapPreparation::retry_close(
  self : FailedDeviceWorkerBootstrapPreparation,
) -> Unit raise DeviceWorkerBootstrapError {
  if self.closed {
    return
  }
  let slot = Ref(self.worker)
  let (worker_failure, model_failure, kernel_failure) = cleanup_all(
    self.roots,
    slot,
  )
  self.worker = slot.val
  self.worker_cleanup_failure = worker_failure
  self.model_cleanup_failure = model_failure
  self.kernel_cleanup_failure = kernel_failure
  match (worker_failure, model_failure, kernel_failure) {
    (None, None, None) => self.closed = true
    (Some(error), _, _) | (_, Some(error), _) | (_, _, Some(error)) =>
      raise error
  }
}

///|
pub fn DeviceWorkerBootstrapOwner::close(
  self : DeviceWorkerBootstrapOwner,
) -> Unit raise DeviceWorkerBootstrapError {
  match self.worker {
    Legacy(owner) => owner.close() catch { error => raise Worker(error) }
    I8(owner) => owner.close() catch { error => raise I8Worker(error) }
    Fp8(owner) => owner.close() catch { error => raise Fp8Worker(error) }
  }
}

///|
pub fn DeviceWorkerBootstrapOwner::readiness_contract(
  self : DeviceWorkerBootstrapOwner,
) -> @worker_wire.WorkerStartupContract raise DeviceWorkerBootstrapError {
  match self.worker {
    Legacy(owner) =>
      owner.readiness_contract() catch {
        error => raise Worker(error)
      }
    I8(owner) =>
      owner.readiness_contract() catch {
        error => raise I8Worker(error)
      }
    Fp8(owner) =>
      owner.readiness_contract() catch {
        error => raise Fp8Worker(error)
      }
  }
}

///|
pub fn DeviceWorkerBootstrapOwner::execute_frame(
  self : DeviceWorkerBootstrapOwner,
  plan : @worker_wire.ValidatedPlanFrame,
  completion : @worker_wire.CompletionFrameWriter,
) -> @worker_wire.ValidatedCompletionFrame raise DeviceWorkerBootstrapError {
  match self.worker {
    Legacy(owner) =>
      owner.execute_frame(plan, completion) catch {
        error => raise Worker(error)
      }
    I8(owner) =>
      owner.execute_frame(plan, completion) catch {
        error => raise I8Worker(error)
      }
    Fp8(owner) =>
      owner.execute_frame(plan, completion) catch {
        error => raise Fp8Worker(error)
      }
  }
}
