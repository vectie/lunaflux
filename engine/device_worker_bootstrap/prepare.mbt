///|
fn acquire_roots_with(
  acquire : (BootstrapRootRole) -> @approved_fs.ApprovedRoot raise DeviceWorkerBootstrapError,
) -> BootstrapRootAcquisition raise DeviceWorkerBootstrapError {
  let roots : BootstrapRoots = { model: None, kernel: None }
  let model = acquire(ModelRoot) catch {
    error => return RootCleanupRequired(preparation_failed(roots, None, error))
  }
  roots.model = Some(model)
  let kernel = acquire(KernelRoot) catch {
    error => return RootCleanupRequired(preparation_failed(roots, None, error))
  }
  roots.kernel = Some(kernel)
  RootsAcquired(roots)
}

///|
fn load_execution(
  source : @worker_wire.EncodedBootstrapSource,
  kernel_root : @approved_fs.ApprovedRoot,
  derived : DerivedBootstrap,
) -> @execution_manifest_file.PagedExecutionAdmission raise DeviceWorkerBootstrapError {
  let kernel = source.kernel()
  let locator = @approved_fs.ApprovedRelativeLocator::new(
    kernel.manifest_locator(),
  ) catch {
    _ => raise Invalid(Locator)
  }
  let digest = @execution_manifest_file.ExecutionManifestDigest::from_sha256(
    kernel.manifest_digest(),
  ) catch {
    _ => raise Invalid(Digest)
  }
  @execution_manifest_file.load_paged(
    kernel_root,
    locator,
    digest,
    derived.plan,
    derived.weights.layout(),
    derived.kv_layout,
    derived.target,
    derived.step_limits,
    derived.execution_limits,
  ) catch {
    error => raise ExecutionManifest(error)
  }
}

///|
fn prepare_with_roots(
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  roots : BootstrapRoots,
) -> DeviceWorkerBootstrapPreparation raise DeviceWorkerBootstrapError {
  let model_root = match roots.model {
    Some(root) => root
    None =>
      return preparation_failed(
        roots,
        None,
        Root(role=ModelRoot, issue=Unavailable),
      )
  }
  let kernel_root = match roots.kernel {
    Some(root) => root
    None =>
      return preparation_failed(
        roots,
        None,
        Root(role=KernelRoot, issue=Unavailable),
      )
  }
  let derived = inspect_model_inputs(startup, source, model_root) catch {
    error => return preparation_failed(roots, None, error)
  }
  let execution = load_execution(source, kernel_root, derived) catch {
    error => return preparation_failed(roots, None, error)
  }
  let worker_plan = @device_worker.admit_plan(
    startup~,
    model=derived.model,
    weight_inspection=derived.weights,
    execution~,
    bootstrap_limits=derived.bootstrap_limits,
  ) catch {
    error => return preparation_failed(roots, None, Worker(error))
  }
  orchestrate_worker_transition(
    startup,
    () => close_kernel_root(roots),
    () => {
      let prepared = @device_worker.prepare(startup, model_root, worker_plan) catch {
        error => raise Worker(error)
      }
      match prepared {
        DeviceWorkerReady(owner) => Ready(owner)
        DeviceWorkerCleanupRequired(failed) => Failed(failed)
      }
    },
    authority => {
      match authority {
        Failed(failed) => Some(Worker(failed.cause()))
        Ready(_) => None
      }
    },
    () => close_model_root(roots),
    authority => {
      match authority {
        Ready(owner) =>
          owner.readiness_contract() catch {
            error => raise Worker(error)
          }
        Failed(failed) => raise Worker(failed.cause())
      }
    },
    (authority, error) => preparation_failed(roots, authority, error),
    authority => {
      match authority {
        Ready(owner) => ChildReady({ worker: owner })
        Failed(_) => abort("failed worker passed readiness barrier")
      }
    },
  )
}

///|
/// Authenticates one received canonical bootstrap source before acquiring the
/// fixed inherited model and kernel root roles. It writes no transport frame.
pub fn prepare(
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
) -> DeviceWorkerBootstrapPreparation raise DeviceWorkerBootstrapError {
  validate_source_envelope(startup, source)
  let acquired = acquire_roots_with(role => {
    let inherited = match role {
      ModelRoot => @approved_fs.Model
      KernelRoot => Kernel
    }
    @approved_fs.ApprovedRoot::open_inherited(inherited) catch {
      error => raise root_error(error, role)
    }
  })
  match acquired {
    RootsAcquired(roots) => prepare_with_roots(startup, source, roots)
    RootCleanupRequired(preparation) => preparation
  }
}
