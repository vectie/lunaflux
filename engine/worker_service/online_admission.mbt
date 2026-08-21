///|
/// Samples the retained monotonic clock immediately before scheduler mutation.
pub fn OnlineWorkerLease::admit(
  self : OnlineWorkerLease,
  request : @core.TokenizedRequest,
) -> Unit raise WorkerServiceError {
  let service = self.require_current()
  if self.request_phase != OnlineRequestEmpty ||
    self.prepared_admission is Some(_) ||
    self.semantic_retention.is_live() {
    raise Invalid(Lifecycle)
  }
  let prepared = service.scheduler.prepare_admission(request)
  prepared.retain_stop_tokens_into(self.semantic_retention) catch {
    error => raise scheduler_error(error)
  }
  let now = self.clock.now_millis() catch {
    _ => {
      self.release_semantic_retention_if_present()
      raise Invalid(Clock)
    }
  }
  service.scheduler.commit_admission(prepared, now) catch {
    error => {
      self.release_semantic_retention_if_present()
      raise scheduler_error(error)
    }
  }
  let handle = try! prepared.request_handle()
  self.handle = Some(handle)
  self.request_generation = handle.request_generation()
  self.request_phase = OnlineRequestLive
}

///|
/// Commits the exact shell preallocated before rooted worker activation.
pub fn OnlineWorkerLease::commit_prepared_admission(
  self : OnlineWorkerLease,
) -> OnlinePreparedAdmissionCommit {
  let service = try! self.require_current()
  if self.request_phase != OnlineRequestEmpty {
    abort("prepared online admission phase invariant")
  }
  let prepared = match self.prepared_admission {
    Some(value) => value
    None => abort("prepared online admission owner invariant")
  }
  try prepared.retain_stop_tokens_into(self.semantic_retention) catch {
    _ => {
      service.scheduler.abort_exclusive_admission(prepared)
      self.prepared_admission = None
      return OnlineAdmissionInvalidated
    }
  } noraise {
    _ => ()
  }
  if self.prepared_clock_read.read_status() == MonotonicReadFailed {
    service.scheduler.abort_exclusive_admission(prepared)
    self.release_semantic_retention_if_present()
    self.prepared_admission = None
    return OnlineAdmissionClockFailed
  }
  match
    service.scheduler.commit_exclusive_admission(
      prepared,
      try! self.prepared_clock_read.value(),
    ) {
    ExclusiveAdmissionExpired => {
      self.release_semantic_retention_if_present()
      self.prepared_admission = None
      return OnlineAdmissionExpired
    }
    ExclusiveAdmissionInvalidated => {
      self.release_semantic_retention_if_present()
      self.prepared_admission = None
      return OnlineAdmissionInvalidated
    }
    ExclusiveAdmissionCommitted => ()
  }
  let handle = prepared.request_handle()
  self.prepared_admission = None
  self.handle = Some(handle)
  self.request_generation = handle.request_generation()
  self.request_phase = OnlineRequestLive
  OnlineAdmissionCommitted
}
