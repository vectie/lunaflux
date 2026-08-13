///|
/// Allocation-sensitive rooted success publication. Both possible shells were
/// allocated before root activation; this selects only the ready owner.
fn publish_ready_root_bound(
  ready : RootBoundWorkerProcessSupervisor,
  failed : FailedRootBoundWorkerProcessStartup,
  process : WorkerProcessSupervisor,
) -> RootBoundWorkerProcessPreparation {
  ready.roots_selected = true
  failed.roots_selected = false
  ready.process = process
  ReadyRootBoundProcess(ready)
}

///|
/// Allocation-sensitive live cleanup publication. It stores only raw status
/// and already-owned references; typed diagnostic values are deferred to
/// cleanup-owner queries.
fn publish_live_root_bound_cleanup(
  ready : RootBoundWorkerProcessSupervisor,
  failed : FailedRootBoundWorkerProcessStartup,
  child : FailedWorkerProcessStartup,
  roots_live : Bool,
  roots_status : Int,
) -> RootBoundWorkerProcessPreparation {
  ready.roots_selected = false
  failed.roots_selected = roots_live
  failed.child = Some(child)
  failed.roots_cleanup_status = roots_status
  failed.closed = false
  RootBoundCleanupRequired(failed)
}
