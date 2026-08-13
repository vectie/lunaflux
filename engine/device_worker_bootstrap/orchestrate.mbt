///|
fn[T] orchestrate_worker_transition(
  startup : @worker_wire.WorkerStartupContract,
  close_kernel : () -> Unit raise DeviceWorkerBootstrapError,
  prepare_worker : () -> T raise DeviceWorkerBootstrapError,
  worker_failure : (T) -> DeviceWorkerBootstrapError?,
  close_model : () -> Unit raise DeviceWorkerBootstrapError,
  readiness : (T) -> @worker_wire.WorkerStartupContract raise DeviceWorkerBootstrapError,
  fail : (T?, DeviceWorkerBootstrapError) -> DeviceWorkerBootstrapPreparation raise DeviceWorkerBootstrapError,
  ready : (T) -> DeviceWorkerBootstrapPreparation,
) -> DeviceWorkerBootstrapPreparation raise DeviceWorkerBootstrapError {
  close_kernel() catch {
    error => return fail(None, error)
  }
  let worker = prepare_worker() catch { error => return fail(None, error) }
  match worker_failure(worker) {
    Some(error) => return fail(Some(worker), error)
    None => ()
  }
  close_model() catch {
    error => return fail(Some(worker), error)
  }
  let actual = readiness(worker) catch {
    error => return fail(Some(worker), error)
  }
  if actual != startup {
    return fail(Some(worker), Invalid(ReadinessContract))
  }
  ready(worker)
}
