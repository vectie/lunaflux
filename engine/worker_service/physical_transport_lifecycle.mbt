///|
fn WorkerPhysicalTransport::begin_recovery(
  self : WorkerPhysicalTransport,
) -> Unit raise WorkerServiceError {
  match self {
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(process) =>
      process.begin_recovery() catch {
        error => raise physical_cleanup_error(error)
      }
    TensorParallelGroup(group) => {
      ignore(take_tensor_parallel_worker_failure(group))
      group.begin_recovery() catch {
        error => raise tensor_parallel_transport_error(error)
      }
    }
  }
}

///|
fn WorkerPhysicalTransport::validate_recovery_binding(
  self : WorkerPhysicalTransport,
  scheduler : @core.Scheduler,
  identity : WorkerServiceIdentity,
  physical_binding : WorkerPhysicalBinding,
) -> Unit raise WorkerServiceError {
  match self {
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(process) => {
      guard physical_binding is RootedSingleBinding(binding) else {
        raise Invalid(Identity)
      }
      let predecessor = scheduler.last_retired_plan_sequence()
      if scheduler.last_submitted_plan_sequence() != predecessor ||
        scheduler.model_identity() != identity.model_identity ||
        scheduler.model_plan_generation().value() !=
        identity.model_generation_value ||
        scheduler.worker_protocol_limits() != identity.worker_limits {
        raise Invalid(Identity)
      }
      process.validate_recovery_binding(
        identity.model_identity,
        binding.bootstrap_digest(),
        identity.bootstrap_source_digest,
        binding.device_ordinal(),
        identity.model_generation_value,
        identity.worker_limits,
        identity.inference_limits,
        predecessor,
      ) catch {
        error => raise physical_transport_error(error)
      }
    }
    TensorParallelGroup(group) => {
      guard physical_binding is TensorParallelBinding else {
        raise Invalid(Identity)
      }
      group.validate_service_identity(
        model_identity=identity.model_identity,
        model_generation_value=identity.model_generation_value,
        bootstrap_source_digest=identity.bootstrap_source_digest,
        worker_limits=identity.worker_limits,
        inference_limits=identity.inference_limits,
        previous_plan_sequence_value=scheduler.last_retired_plan_sequence(),
      ) catch {
        error => raise tensor_parallel_transport_error(error)
      }
    }
  }
}

///|
fn WorkerPhysicalTransport::restart(
  self : WorkerPhysicalTransport,
) -> WorkerPhysicalRestart raise WorkerServiceError {
  match self {
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(process) => {
      let restart = process.restart() catch {
        error => raise physical_transport_error(error)
      }
      match restart {
        RootBoundRestarted => PhysicalRestarted
        RootBoundRestartCleanupRequired => PhysicalRestartCleanupRequired
      }
    }
    TensorParallelGroup(group) => {
      let result = group.start() catch {
        error => raise tensor_parallel_transport_error(error)
      }
      match result {
        GroupReady => PhysicalRestarted
        GroupCleanupRequired => {
          if group.has_failure_publication_obligation() {
            ignore(take_tensor_parallel_worker_failure(group))
          }
          PhysicalRestartCleanupRequired
        }
      }
    }
  }
}

///|
fn WorkerPhysicalTransport::retry_replacement_cleanup(
  self : WorkerPhysicalTransport,
) -> Unit raise WorkerServiceError {
  match self {
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(process) =>
      process.retry_replacement_cleanup() catch {
        error => raise physical_cleanup_error(error)
      }
    TensorParallelGroup(group) => {
      let progress = group.progress_cleanup() catch {
        error => raise tensor_parallel_transport_error(error)
      }
      match progress {
        GroupReplacementReady => ()
        GroupCleanupAdvanced | GroupCleanupPending | GroupCleanupStuck(_) =>
          raise PhysicalTransport(PhysicalBackpressured)
        GroupCloseComplete => raise Invalid(Recovery)
      }
    }
  }
}

///|
fn WorkerPhysicalTransport::shutdown_clean(
  self : WorkerPhysicalTransport,
) -> Unit raise WorkerServiceError {
  match self {
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(process) =>
      process.shutdown_clean() catch {
        error => raise physical_transport_error(error)
      }
    TensorParallelGroup(group) => {
      if group.lifecycle() == Ready || group.lifecycle() == Prepared {
        group.begin_close() catch {
          error => raise tensor_parallel_transport_error(error)
        }
      } else if group.lifecycle() == Closing {
        ignore(
          group.progress_cleanup() catch {
            error => raise tensor_parallel_transport_error(error)
          },
        )
      }
      if group.lifecycle() != Closed {
        raise PhysicalTransport(PhysicalBackpressured)
      }
    }
  }
}

///|
fn WorkerPhysicalTransport::close(
  self : WorkerPhysicalTransport,
) -> Unit raise WorkerServiceError {
  match self {
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(process) =>
      process.close() catch {
        error => raise physical_cleanup_error(error)
      }
    TensorParallelGroup(group) => {
      if group.lifecycle() == Ready || group.lifecycle() == Prepared {
        group.begin_close() catch {
          error => raise tensor_parallel_transport_error(error)
        }
      } else if group.lifecycle() == Closing {
        ignore(
          group.progress_cleanup() catch {
            error => raise tensor_parallel_transport_error(error)
          },
        )
      }
      if group.lifecycle() != Closed {
        raise PhysicalTransport(PhysicalBackpressured)
      }
    }
  }
}

///|
fn WorkerPhysicalTransport::begin_recovery_maintenance(
  self : WorkerPhysicalTransport,
  fallback_failure : @worker_protocol.WorkerFailure,
) -> @worker_protocol.WorkerFailure raise WorkerServiceError {
  match self {
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(process) => {
      process.begin_recovery_maintenance() catch {
        error => raise physical_cleanup_error(error)
      }
      fallback_failure
    }
    TensorParallelGroup(group) => {
      let failure = take_tensor_parallel_worker_failure(group)
      group.begin_recovery() catch {
        error => raise tensor_parallel_transport_error(error)
      }
      failure
    }
  }
}

///|
fn WorkerPhysicalTransport::begin_close_maintenance(
  self : WorkerPhysicalTransport,
) -> Unit raise WorkerServiceError {
  match self {
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(process) =>
      process.begin_close_maintenance() catch {
        error => raise physical_cleanup_error(error)
      }
    TensorParallelGroup(group) =>
      group.begin_close() catch {
        error => raise tensor_parallel_transport_error(error)
      }
  }
}

///|
fn WorkerPhysicalTransport::begin_replacement_cleanup_maintenance(
  self : WorkerPhysicalTransport,
) -> Unit raise WorkerServiceError {
  match self {
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(process) =>
      process.begin_replacement_cleanup_maintenance() catch {
        error => raise physical_cleanup_error(error)
      }
    TensorParallelGroup(group) =>
      if group.lifecycle() == Faulted {
        ignore(
          group.take_group_failure() catch {
            error => raise tensor_parallel_transport_error(error)
          },
        )
        group.begin_recovery() catch {
          error => raise tensor_parallel_transport_error(error)
        }
      } else if group.lifecycle() != Recovering {
        raise Invalid(Recovery)
      }
  }
}

///|
fn WorkerPhysicalTransport::progress_maintenance(
  self : WorkerPhysicalTransport,
) -> WorkerPhysicalMaintenanceProgress raise WorkerServiceError {
  match self {
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(process) => {
      let progress = process.progress_maintenance() catch {
        error => raise physical_cleanup_error(error)
      }
      let code = if progress.is_cleanup_required() {
        1
      } else if progress.is_cleanup_stuck() {
        2
      } else if progress.is_pending() {
        3
      } else if progress.is_advanced() {
        4
      } else if progress.is_complete() {
        5
      } else {
        0
      }
      WorkerPhysicalMaintenanceProgress(code)
    }
    TensorParallelGroup(group) => {
      let progress = group.progress_cleanup() catch {
        error => raise tensor_parallel_transport_error(error)
      }
      WorkerPhysicalMaintenanceProgress(
        match progress {
          GroupCleanupStuck(_) => 2
          GroupCleanupPending => 3
          GroupCleanupAdvanced => 4
          GroupReplacementReady | GroupCloseComplete => 5
        },
      )
    }
  }
}

///|
fn WorkerPhysicalMaintenanceProgress::is_cleanup_required(
  self : WorkerPhysicalMaintenanceProgress,
) -> Bool {
  self.0 == 1
}

///|
fn WorkerPhysicalMaintenanceProgress::is_cleanup_stuck(
  self : WorkerPhysicalMaintenanceProgress,
) -> Bool {
  self.0 == 2
}

///|
fn WorkerPhysicalMaintenanceProgress::is_pending(
  self : WorkerPhysicalMaintenanceProgress,
) -> Bool {
  self.0 == 3
}

///|
fn WorkerPhysicalMaintenanceProgress::is_advanced(
  self : WorkerPhysicalMaintenanceProgress,
) -> Bool {
  self.0 == 4
}

///|
fn WorkerPhysicalMaintenanceProgress::is_complete(
  self : WorkerPhysicalMaintenanceProgress,
) -> Bool {
  self.0 == 5
}

///|
fn WorkerPhysicalTransport::maintenance_remaining_millis(
  self : WorkerPhysicalTransport,
) -> Int raise WorkerServiceError {
  match self {
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(process) =>
      process.maintenance_remaining_millis() catch {
        error => raise physical_cleanup_error(error)
      }
    TensorParallelGroup(_) => 0
  }
}
