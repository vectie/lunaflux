///|
fn load_qwen3_execution(
  source : @worker_wire.EncodedBootstrapSource,
  kernel_root : @approved_fs.ApprovedRoot,
  derived : DerivedQwen3Bootstrap,
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
fn prepare_qwen3_with_roots(
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  roots : BootstrapRoots,
) -> DeviceWorkerBootstrapPreparation raise DeviceWorkerBootstrapError {
  guard roots.model is Some(model_root) && roots.kernel is Some(kernel_root) else {
    return preparation_failed(
      roots,
      None,
      Root(role=ModelRoot, issue=Unavailable),
    )
  }
  let derived = derive_qwen3_bootstrap(startup, source, model_root) catch {
    error => return preparation_failed(roots, None, error)
  }
  let execution = load_qwen3_execution(source, kernel_root, derived) catch {
    error => return preparation_failed(roots, None, error)
  }
  let fused_runtime = load_optional_fused_runtime(source, kernel_root) catch {
    error => return preparation_failed(roots, None, error)
  }
  let sampling_policy = @device_worker.admit_bootstrap_sampling_policy(
    source,
    derived.plan,
    execution,
  ) catch {
    error => return preparation_failed(roots, None, Worker(error))
  }
  let worker_plan = @device_worker.admit_numeric_bf16_plan_v1(
    startup~,
    model_plan=derived.plan,
    weight_inspection=derived.weights,
    execution~,
    bootstrap_limits=derived.bootstrap_limits,
    sampling_policy~,
    fused_catalog=match fused_runtime {
      Some(runtime) => runtime.catalog
      None => None
    },
    fused_bundle=match fused_runtime {
      Some(runtime) => runtime.bundle
      None => None
    },
    reusable_fused_residual_runtime=match fused_runtime {
      Some(runtime) => runtime.reusable_residual
      None => None
    },
    reusable_fused_runtime_bundle=match fused_runtime {
      Some(runtime) => runtime.reusable_bundle
      None => None
    },
    fused_qkv_readonly_source=match fused_runtime {
      Some(runtime) => runtime.production
      None => None
    },
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
        DeviceWorkerReady(owner) => LegacyReady(owner)
        DeviceWorkerCleanupRequired(failed) => LegacyFailed(failed)
      }
    },
    authority => {
      match authority {
        LegacyFailed(failed) => Some(Worker(failed.cause()))
        LegacyReady(_) => None
        _ => Some(Invalid(Recipe))
      }
    },
    () => close_model_root(roots),
    authority => {
      match authority {
        LegacyReady(owner) =>
          owner.readiness_contract() catch {
            error => raise Worker(error)
          }
        LegacyFailed(failed) => raise Worker(failed.cause())
        _ => raise Invalid(Recipe)
      }
    },
    (authority, error) => preparation_failed(roots, authority, error),
    authority => {
      match authority {
        LegacyReady(owner) => ChildReady({ worker: Legacy(owner) })
        _ => abort("invalid Qwen3 worker passed readiness barrier")
      }
    },
  )
}

///|
fn prepare_qwen3(
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
) -> DeviceWorkerBootstrapPreparation raise DeviceWorkerBootstrapError {
  let acquired = acquire_roots_with(role => {
    @approved_fs.ApprovedRoot::open_inherited(
      match role {
        ModelRoot => Model
        KernelRoot => Kernel
      },
    ) catch {
      error => raise root_error(error, role)
    }
  })
  match acquired {
    RootsAcquired(roots) => prepare_qwen3_with_roots(startup, source, roots)
    RootCleanupRequired(preparation) => preparation
  }
}
