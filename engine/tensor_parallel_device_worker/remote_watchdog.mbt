///|
priv struct RemoteWorkerWatchdog {
  clock : @monotonic_clock.ReusableMonotonicRead
  mut lease : @remote_lease.RemoteLease
}

///|
fn prepare_remote_watchdog(
  remote : @remote_startup.RemoteRankStartup?,
) -> RemoteWorkerWatchdog? raise TensorParallelDeviceWorkerError {
  guard remote is Some(startup) else { return None }
  let clock = @monotonic_clock.MonotonicClock::system().prepare_poller()
  if !clock.poll() {
    raise Invalid(RemoteLease)
  }
  let now = clock.value() catch { _ => raise Invalid(RemoteLease) }
  let lease = startup.start_lease(now) catch { _ => raise Invalid(RemoteLease) }
  Some({ clock, lease, })
}

///|
fn poll_watchdog(watchdog : RemoteWorkerWatchdog) -> Unit {
  if !watchdog.clock.poll() {
    watchdog.lease = watchdog.lease.transport_lost()
    return
  }
  let now = watchdog.clock.value() catch {
    _ => {
      watchdog.lease = watchdog.lease.transport_lost()
      return
    }
  }
  watchdog.lease = watchdog.lease.poll(now)
}

///|
fn TensorParallelDeviceWorkerOwner::enforce_remote_lease(
  self : TensorParallelDeviceWorkerOwner,
  watchdog : RemoteWorkerWatchdog,
) -> Unit raise TensorParallelDeviceWorkerError {
  if watchdog.lease.status() == Active {
    return
  }
  if self.primary_failure is None {
    self.primary_failure = Some(Invalid(RemoteLease))
    self.failure_class = Some(CollectiveFailure)
  }
  // Expiration aborts even an idle Ready communicator. Cleanup failure retains
  // the existing owner and its explicit retry authority; it never revives a lease.
  self.resources.collective.abort_required = true
  self.resources.executor.abort_required = true
  self.close()
  raise Invalid(RemoteLease)
}

///|
/// The remote control loop must call this on idle turns as well as pending GPU
/// turns. Every execution entry point also calls it before doing work.
pub fn TensorParallelDeviceWorkerOwner::poll_remote_lease(
  self : TensorParallelDeviceWorkerOwner,
) -> Unit raise TensorParallelDeviceWorkerError {
  guard self.watchdog is Some(watchdog) else { return }
  poll_watchdog(watchdog)
  self.enforce_remote_lease(watchdog)
}

///|
/// Only a heartbeat from the admitted channel may call this. No peer timestamp
/// is accepted and stale or replayed sequences permanently fence the worker.
pub fn TensorParallelDeviceWorkerOwner::renew_remote_lease(
  self : TensorParallelDeviceWorkerOwner,
  generation~ : UInt64,
  sequence~ : UInt64,
) -> Unit raise TensorParallelDeviceWorkerError {
  guard self.watchdog is Some(watchdog) else { raise Invalid(ResourceState) }
  if self.lifecycle == WorkerClosed || self.lifecycle == WorkerClosingFailed {
    raise Invalid(ResourceState)
  }
  poll_watchdog(watchdog)
  let now = watchdog.clock.value() catch {
    _ => {
      watchdog.lease = watchdog.lease.transport_lost()
      self.enforce_remote_lease(watchdog)
      return
    }
  }
  watchdog.lease = watchdog.lease.renew(
    generation~,
    sequence~,
    local_now_millis=now,
  )
  self.enforce_remote_lease(watchdog)
}

///|
pub fn TensorParallelDeviceWorkerOwner::remote_transport_lost(
  self : TensorParallelDeviceWorkerOwner,
) -> Unit raise TensorParallelDeviceWorkerError {
  guard self.watchdog is Some(watchdog) else { raise Invalid(ResourceState) }
  watchdog.lease = watchdog.lease.transport_lost()
  self.enforce_remote_lease(watchdog)
}

///|
/// Startup-only evidence retained from actual executor admission.
pub fn TensorParallelDeviceWorkerOwner::remote_contract(
  self : TensorParallelDeviceWorkerOwner,
) -> (
  @remote_startup.RemoteRankStartup,
  @tensor_parallel_worker_bootstrap.TensorParallelRankBootstrapEnvelope,
) raise TensorParallelDeviceWorkerError {
  guard self.plan.remote is Some(remote) else { raise Invalid(ResourceState) }
  guard self.plan.remote_envelope is Some(envelope) else {
    raise Invalid(ResourceState)
  }
  (remote, envelope)
}
