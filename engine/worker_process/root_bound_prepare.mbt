///|
/// Acquires a private duplicate pair from caller-owned roots and uses it for
/// the initial child and every replacement. The input roots are only borrowed;
/// no public API accepts or returns the retained pair.
fn preflight_root_bound_executable(
  executable : @worker_executable_file.WorkerExecutableAdmission,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  limits : WorkerProcessLimits,
  parent_approval : @worker_wire.WorkerParentApprovalSubjects?,
) -> PreparedWorkerProcessStartup raise RootBoundWorkerProcessError {
  preflight_prepare_approved(
    executable, startup, source, limits, parent_approval,
  ) catch {
    error => raise RootBoundFailure(Process(error))
  }
}

///|
fn prepare_root_bound_with_authority(
  executable : @worker_executable_file.WorkerExecutableAdmission,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  limits : WorkerProcessLimits,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  exchange_claimed : Bool,
  expected_predecessor : UInt64,
  parent_approval : @worker_wire.WorkerParentApprovalSubjects?,
) -> RootBoundWorkerProcessPreparation raise RootBoundWorkerProcessError {
  if source.digest() != startup.bootstrap_source_digest() {
    raise RootBoundFailure(Process(Invalid(SourceMismatch)))
  }
  if exchange_claimed &&
    (
      startup.has_predecessor() != (expected_predecessor != 0UL) ||
      startup.predecessor_value() != expected_predecessor
    ) {
    raise RootBoundFailure(Process(Invalid(Sequence)))
  }
  validate_preflight_envelope(startup, source, limits) catch {
    error => raise RootBoundFailure(Process(error))
  }
  if !executable.is_activation_live() {
    raise RootBoundFailure(Process(Transport(SpawnFailed)))
  }
  let labels = source.roots()
  require_model_root_binding(model_root, labels.model_root())
  require_kernel_root_binding(kernel_root, labels.kernel_root())
  let prepared = preflight_root_bound_executable(
    executable, startup, source, limits, parent_approval,
  )
  let prepared_roots = @approved_fs.prepare_worker_approved_roots()
  let roots = prepared_roots.root_owner()
  let ready_owner : RootBoundWorkerProcessSupervisor = {
    approved_executable: prepared.approved_executable,
    limits,
    roots,
    roots_selected: false,
    process: prepared.supervisor,
    failed_replacement: None,
    cause: None,
    child_cleanup_failure: None,
    roots_cleanup_failure: None,
    transport_mode: if exchange_claimed {
      2
    } else {
      0
    },
    exchange_epoch: 0UL,
    active_exchange_epoch: 0UL,
    exchange_phase: -1,
    exchange_side: -1,
    exchange_sequence: 0UL,
    exchange_plan: @worker_protocol.SubmittedSchedulePlanSlot::new(),
    maintenance_mode: 0,
    lifecycle: RootBoundReady,
  }
  let failed_owner : FailedRootBoundWorkerProcessStartup = {
    approved_executable: prepared.approved_executable,
    roots,
    roots_selected: false,
    child: None,
    child_cleanup_process: None,
    child_cleanup_native: None,
    roots_cleanup_status: 0,
    closed: false,
  }
  let root_activation_status = @approved_fs.acquire_prepared_worker_approved_roots_status(
    prepared_roots, model_root, kernel_root,
  )
  if root_activation_status != 0 {
    raise RootBoundFailure(
      if root_activation_status == 7 {
        RootsBusy
      } else {
        RootsCloseFailed
      },
    )
  }
  let startup_result = prepare_preflighted(source, limits, prepared, roots)
  match startup_result {
    ReadyProcess(process) =>
      publish_ready_root_bound(ready_owner, failed_owner, process)
    CleanupRequired(child) => {
      let roots_status = close_root_pair_status(roots)
      let roots_live = roots_still_live(roots)
      if !child.is_closed() || roots_live {
        return publish_live_root_bound_cleanup(
          ready_owner, failed_owner, child, roots_live, roots_status,
        )
      }
      ready_owner.roots_selected = false
      failed_owner.roots_selected = false
      failed_owner.child = Some(child)
      failed_owner.roots_cleanup_status = roots_status
      failed_owner.closed = !approved_executable_live(
        failed_owner.approved_executable,
      )
      // Root acquisition has happened. Even when every acquired authority is
      // already terminally closed, publish the explicit cleanup outcome so no
      // caller can confuse this with a pre-acquisition raised rejection.
      RootBoundCleanupRequired(failed_owner)
    }
  }
}
