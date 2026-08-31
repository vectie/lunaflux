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
  luna_parent_witness : @luna_capability_manifest.LunaParentStartupApprovalWitness?,
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
    luna_approval=kernel.luna_approval(),
    luna_parent_witness~,
  ) catch {
    error => raise ExecutionManifest(error)
  }
}

///|
fn prepare_with_roots(
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  roots : BootstrapRoots,
  luna_parent_witness : @luna_capability_manifest.LunaParentStartupApprovalWitness?,
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
  let execution = load_execution(
    source, kernel_root, derived, luna_parent_witness,
  ) catch {
    error => return preparation_failed(roots, None, error)
  }
  let fused_runtime = load_optional_fused_runtime(source, kernel_root) catch {
    error => return preparation_failed(roots, None, error)
  }
  reject_reusable_qwen_bundle_for_non_qwen(fused_runtime) catch {
    error => return preparation_failed(roots, None, error)
  }
  let sampling_policy = @device_worker.admit_bootstrap_sampling_policy(
    source,
    derived.plan,
    execution,
  ) catch {
    error => return preparation_failed(roots, None, Worker(error))
  }
  let worker_plan = @device_worker.admit_plan(
    startup~,
    model=derived.model,
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
        _ => abort("invalid worker passed readiness barrier")
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
  luna_parent_witness? : @luna_capability_manifest.LunaParentStartupApprovalWitness? = None,
) -> DeviceWorkerBootstrapPreparation raise DeviceWorkerBootstrapError {
  match (source.kernel().luna_approval(), luna_parent_witness) {
    (None, None) => ()
    (Some(_), Some(witness)) =>
      witness.require_launch_identity(
        bootstrap_digest_sha256=startup.bootstrap_digest().as_hex(),
        source_digest_sha256=startup.bootstrap_source_digest().as_hex(),
        model_generation=startup.model_generation_value(),
        device_ordinal=startup.device_ordinal(),
      ) catch {
        _ => raise Invalid(Recipe)
      }
    _ => raise Invalid(Recipe)
  }
  if source.recipe() == DenseLlamaFp8ReusablePagedAotApprovedV9 {
    return prepare_fp8(startup, source, luna_parent_witness)
  }
  if source.recipe() == DenseLlamaI8PagedAotV6 {
    return prepare_i8(startup, source)
  }
  if source.recipe() == DenseMistralBf16PagedAotV7 ||
    source.recipe() == DenseMistralBf16PagedAotExtensionsV11 {
    return prepare_mistral(startup, source)
  }
  if source.recipe() == DenseQwen3Bf16PagedAotV12 ||
    source.recipe() == DenseQwen3Bf16PagedAotExtensionsV13 {
    return prepare_qwen3(startup, source)
  }
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
    RootsAcquired(roots) =>
      prepare_with_roots(startup, source, roots, luna_parent_witness)
    RootCleanupRequired(preparation) => preparation
  }
}
