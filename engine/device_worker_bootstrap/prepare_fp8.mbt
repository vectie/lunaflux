///|
fn load_fp8_execution(
  source : @worker_wire.EncodedBootstrapSource,
  kernel_root : @approved_fs.ApprovedRoot,
  derived : DerivedFp8Bootstrap,
  luna_parent_witness : @luna_capability_manifest.LunaParentStartupApprovalWitness?,
) -> @execution_manifest_file.Fp8PagedExecutionAdmissionV5 raise DeviceWorkerBootstrapError {
  guard source.kernel().luna_approval() is Some(approval) else {
    raise Invalid(Recipe)
  }
  let locator = @approved_fs.ApprovedRelativeLocator::new(
    source.kernel().manifest_locator(),
  ) catch {
    _ => raise Invalid(Locator)
  }
  let digest = @execution_manifest_file.ExecutionManifestDigest::from_sha256(
    source.kernel().manifest_digest(),
  ) catch {
    _ => raise Invalid(Digest)
  }
  let execution = @execution_manifest_file.load_paged_fp8_v5(
    kernel_root,
    locator,
    digest,
    derived.plan,
    derived.inspection,
    derived.kv_layout,
    derived.numeric,
    derived.target,
    derived.step_limits,
    derived.manifest_limits,
    derived.fp8_limits,
    derived.startup_limits,
    derived.exact_limits,
    derived.reusable_limits,
    approval,
    derived.release_limits,
    parent_witness=luna_parent_witness,
  ) catch {
    error => raise ExecutionManifest(error)
  }
  guard source.fp8_execution() is Some(source_execution) else {
    raise Invalid(Recipe)
  }
  if execution.reusable_recipe().digest().as_hex() !=
    source_execution.reusable_recipe_digest() {
    raise Invalid(ReleaseIdentity)
  }
  execution
}

///|
fn prepare_fp8_with_roots(
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  roots : BootstrapRoots,
  luna_parent_witness : @luna_capability_manifest.LunaParentStartupApprovalWitness?,
) -> DeviceWorkerBootstrapPreparation raise DeviceWorkerBootstrapError {
  guard roots.model is Some(model_root) && roots.kernel is Some(kernel_root) else {
    return preparation_failed(
      roots,
      None,
      Root(role=ModelRoot, issue=Unavailable),
    )
  }
  let derived = inspect_fp8_model_inputs(startup, source, model_root) catch {
    error => return preparation_failed(roots, None, error)
  }
  let execution = load_fp8_execution(
    source, kernel_root, derived, luna_parent_witness,
  ) catch {
    error => return preparation_failed(roots, None, error)
  }
  let plan = @fp8_device_worker_v3.admit_plan_v3(
    startup,
    source,
    execution,
    derived.expected_total_memory_bytes,
  ) catch {
    error => return preparation_failed(roots, None, Fp8Worker(error))
  }
  orchestrate_worker_transition(
    startup,
    () => close_kernel_root(roots),
    () => {
      let prepared = @fp8_device_worker_v3.prepare_v3(startup, model_root, plan) catch {
        error => raise Fp8Worker(error)
      }
      match prepared {
        Fp8DeviceWorkerReadyV3(owner) => Fp8Ready(owner)
        Fp8DeviceWorkerCleanupRequiredV3(failed) => Fp8Failed(failed)
      }
    },
    authority => {
      match authority {
        Fp8Failed(failed) => Some(Fp8Worker(failed.cause()))
        Fp8Ready(_) => None
        _ => Some(Invalid(Recipe))
      }
    },
    () => close_model_root(roots),
    authority => {
      match authority {
        Fp8Ready(owner) =>
          owner.readiness_contract() catch {
            error => raise Fp8Worker(error)
          }
        Fp8Failed(failed) => raise Fp8Worker(failed.cause())
        _ => raise Invalid(Recipe)
      }
    },
    (authority, error) => preparation_failed(roots, authority, error),
    authority => {
      match authority {
        Fp8Ready(owner) => ChildReady({ worker: Fp8(owner) })
        _ => abort("invalid FP8 worker passed readiness barrier")
      }
    },
  )
}

///|
fn prepare_fp8(
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  luna_parent_witness : @luna_capability_manifest.LunaParentStartupApprovalWitness?,
) -> DeviceWorkerBootstrapPreparation raise DeviceWorkerBootstrapError {
  validate_fp8_source_envelope(startup, source)
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
    RootsAcquired(roots) =>
      prepare_fp8_with_roots(startup, source, roots, luna_parent_witness)
    RootCleanupRequired(preparation) => preparation
  }
}
