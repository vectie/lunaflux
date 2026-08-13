///|
fn[T] finish_ready_child(
  expected : @worker_wire.WorkerStartupContract,
  owner : T,
  readiness : (T) -> @worker_wire.WorkerStartupContract raise DeviceWorkerChildError,
  publish_ready : (@worker_wire.WorkerStartupContract) -> Unit raise DeviceWorkerChildError,
  wait_eof : () -> Unit raise DeviceWorkerChildError,
  close_owner : (T) -> Unit raise DeviceWorkerChildError,
) -> Bool {
  let body_succeeded = try {
    let actual = readiness(owner)
    if actual != expected {
      raise StartupFailed
    }
    publish_ready(actual)
    wait_eof()
    true
  } catch {
    _ => false
  }
  let close_succeeded = try {
    close_owner(owner)
    true
  } catch {
    _ => false
  }
  body_succeeded && close_succeeded
}

///|
fn[T, F] run_startup_with(
  read_configure : () -> @worker_wire.WorkerStartupContract raise DeviceWorkerChildError,
  read_source : () -> @worker_wire.EncodedBootstrapSource raise DeviceWorkerChildError,
  prepare : (
    @worker_wire.WorkerStartupContract,
    @worker_wire.EncodedBootstrapSource,
  ) -> ChildBootstrap[T, F] raise DeviceWorkerChildError,
  retry_cleanup : (F) -> Unit raise DeviceWorkerChildError,
  readiness : (T) -> @worker_wire.WorkerStartupContract raise DeviceWorkerChildError,
  publish_ready : (@worker_wire.WorkerStartupContract) -> Unit raise DeviceWorkerChildError,
  wait_eof : () -> Unit raise DeviceWorkerChildError,
  close_owner : (T) -> Unit raise DeviceWorkerChildError,
) -> Bool {
  let startup = read_configure() catch { _ => return false }
  let source = read_source() catch { _ => return false }
  let prepared = prepare(startup, source) catch { _ => return false }
  match prepared {
    BootstrapCleanup(failed) => {
      retry_cleanup(failed) catch {
        _ => ()
      }
      false
    }
    BootstrapReady(owner) =>
      finish_ready_child(
        startup, owner, readiness, publish_ready, wait_eof, close_owner,
      )
  }
}
