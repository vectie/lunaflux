///|
fn[P, W, C] serve_serialized_with(
  read_plan : () -> P raise DeviceWorkerChildError,
  acquire_writer : (P) -> W raise DeviceWorkerChildError,
  execute : (P, W) -> C raise DeviceWorkerChildError,
  publish : (C) -> Unit raise DeviceWorkerChildError,
) -> Unit raise DeviceWorkerChildError {
  for ;; {
    let plan = read_plan() catch {
      CleanEof => return
      _ => raise StartupFailed
    }
    let writer = acquire_writer(plan)
    let completion = execute(plan, writer)
    publish(completion)
  }
}

///|
fn[T, L] finish_ready_child(
  expected : @worker_wire.WorkerStartupContract,
  owner : T,
  readiness : (T) -> @worker_wire.WorkerStartupContract raise DeviceWorkerChildError,
  prepare_loop : (@worker_wire.WorkerStartupContract) -> L raise DeviceWorkerChildError,
  publish_ready : (@worker_wire.WorkerStartupContract) -> Unit raise DeviceWorkerChildError,
  serve : (T, L) -> Unit raise DeviceWorkerChildError,
  close_owner : (T) -> Unit raise DeviceWorkerChildError,
) -> Bool {
  let body_succeeded = try {
    let actual = readiness(owner)
    if actual != expected {
      raise StartupFailed
    }
    let loop_state = prepare_loop(actual)
    publish_ready(actual)
    serve(owner, loop_state)
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
fn[T, F, L] run_startup_with(
  read_configure : () -> @worker_wire.WorkerStartupContract raise DeviceWorkerChildError,
  read_source : () -> @worker_wire.EncodedBootstrapSource raise DeviceWorkerChildError,
  prepare : (
    @worker_wire.WorkerStartupContract,
    @worker_wire.EncodedBootstrapSource,
  ) -> ChildBootstrap[T, F] raise DeviceWorkerChildError,
  retry_cleanup : (F) -> Unit raise DeviceWorkerChildError,
  readiness : (T) -> @worker_wire.WorkerStartupContract raise DeviceWorkerChildError,
  prepare_loop : (@worker_wire.WorkerStartupContract) -> L raise DeviceWorkerChildError,
  publish_ready : (@worker_wire.WorkerStartupContract) -> Unit raise DeviceWorkerChildError,
  serve : (T, L) -> Unit raise DeviceWorkerChildError,
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
        startup, owner, readiness, prepare_loop, publish_ready, serve, close_owner,
      )
  }
}
