///|
fn load_i8_execution(
  source : @worker_wire.EncodedBootstrapSource,
  kernel_root : @approved_fs.ApprovedRoot,
  derived : DerivedI8Bootstrap,
  device_ordinal : Int,
) -> @execution_manifest_file.I8PagedExecutionAdmissionV4 raise DeviceWorkerBootstrapError {
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
  @execution_manifest_file.load_paged_i8_v4(
    kernel_root,
    locator,
    digest,
    derived.plan,
    derived.inspection.layout(),
    derived.kv_layout,
    derived.device_numeric,
    derived.target,
    derived.step_limits,
    device_ordinal~,
    derived.manifest_limits,
    derived.i8_limits,
    derived.bootstrap_limits,
  ) catch {
    error => raise ExecutionManifest(error)
  }
}

///|
fn prepare_i8_with_roots(
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
  let derived = inspect_i8_model_inputs(startup, source, model_root) catch {
    error => return preparation_failed(roots, None, error)
  }
  let execution = load_i8_execution(
    source,
    kernel_root,
    derived,
    startup.device_ordinal(),
  ) catch {
    error => return preparation_failed(roots, None, error)
  }
  let worker_plan = @i8_device_worker.admit_plan(
    startup~,
    model_plan=derived.plan,
    inspection=derived.inspection,
    execution~,
    expected_total_memory_bytes=derived.expected_total_memory_bytes,
  ) catch {
    error => return preparation_failed(roots, None, I8Worker(error))
  }
  orchestrate_worker_transition(
    startup,
    () => close_kernel_root(roots),
    () => {
      let prepared = @i8_device_worker.prepare(startup, model_root, worker_plan) catch {
        error => raise I8Worker(error)
      }
      match prepared {
        I8DeviceWorkerReady(owner) => I8Ready(owner)
        I8DeviceWorkerCleanupRequired(failed) => I8Failed(failed)
      }
    },
    authority => {
      match authority {
        I8Failed(failed) => Some(I8Worker(failed.cause()))
        I8Ready(_) => None
        _ => Some(Invalid(Recipe))
      }
    },
    () => close_model_root(roots),
    authority => {
      match authority {
        I8Ready(owner) =>
          owner.readiness_contract() catch {
            error => raise I8Worker(error)
          }
        I8Failed(failed) => raise I8Worker(failed.cause())
        _ => raise Invalid(Recipe)
      }
    },
    (authority, error) => preparation_failed(roots, authority, error),
    authority => {
      match authority {
        I8Ready(owner) => ChildReady({ worker: I8(owner) })
        _ => abort("invalid I8 worker passed readiness barrier")
      }
    },
  )
}

///|
/// Authenticates the separate v6 I8 source family before acquiring the exact
/// inherited roots. Generic legacy and tensor-parallel sources cannot enter.
fn prepare_i8(
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
) -> DeviceWorkerBootstrapPreparation raise DeviceWorkerBootstrapError {
  validate_i8_source_envelope(startup, source)
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
    RootsAcquired(roots) => prepare_i8_with_roots(startup, source, roots)
    RootCleanupRequired(preparation) => preparation
  }
}
