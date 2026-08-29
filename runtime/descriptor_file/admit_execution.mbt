///|
fn admit_kv_layout(
  plan : @plan.ModelPlan,
  claims : RuntimeDescriptorExecutionClaims,
  ceilings : @worker_wire.WorkerBootstrapCeilings,
) -> @device_layout.DeviceKvLayout raise RuntimeDescriptorFileError {
  let limits = @device_layout.DeviceKvLimits::new(
    tokens_per_page=claims.tokens_per_page,
    total_page_count=claims.total_page_count,
    max_arena_bytes=ceilings.max_kv_arena_bytes(),
  ) catch {
    _ => raise AdmissionRejected(KvLayout)
  }
  @device_layout.plan(plan.identity(), plan.kv_cache_geometry(), limits) catch {
    _ => raise AdmissionRejected(KvLayout)
  }
}

///|
fn admit_step_limits(
  worker : @worker_protocol.WorkerProtocolLimits,
) -> @device_step.DeviceStepLimits raise RuntimeDescriptorFileError {
  @device_step.DeviceStepLimits::new(
    max_rows=worker.max_plan_rows(),
    max_tokens=worker.max_plan_tokens(),
    max_pages=worker.max_plan_pages(),
    max_sequence_tokens=worker.max_sequence_tokens(),
    max_token_id=worker.max_token_id(),
  ) catch {
    _ => raise AdmissionRejected(DeviceStepLimits)
  }
}

///|
fn admit_execution_limits(
  operation_count : Int,
  ceilings : @worker_wire.WorkerBootstrapCeilings,
) -> @execution_manifest_file.ExecutionManifestFileLimits raise RuntimeDescriptorFileError {
  if operation_count <= 0 {
    raise AdmissionRejected(ExecutionLimits)
  }
  let artifacts = @artifact.KernelArtifactLimits::new(
    max_modules=operation_count,
    max_entry_points=operation_count,
    max_module_bytes=ceilings.max_module_bytes(),
    max_total_module_bytes=ceilings.max_total_module_bytes(),
    max_function_symbol_bytes=RECIPE_MAX_SYMBOL_BYTES,
  ) catch {
    _ => raise AdmissionRejected(ExecutionLimits)
  }
  let artifact_files = @artifact_file.KernelArtifactFileLimits::new(
    max_manifest_bytes=ceilings.max_execution_manifest_bytes(),
    max_json_depth=RECIPE_JSON_DEPTH,
    max_path_bytes=4096,
    artifact=artifacts,
  ) catch {
    _ => raise AdmissionRejected(ExecutionLimits)
  }
  let memory = @device_memory.DeviceMemoryLimits::new(
    activation_alignment=RECIPE_ALIGNMENT_BYTES,
    max_arena_bytes=ceilings.max_activation_arena_bytes(),
  ) catch {
    _ => raise AdmissionRejected(ExecutionLimits)
  }
  let contracts = @launch_contract.LaunchContractLimits::new(
    max_profiles=1,
    max_contracts=operation_count,
    max_operands_per_contract=RECIPE_MAX_OPERANDS,
  ) catch {
    _ => raise AdmissionRejected(ExecutionLimits)
  }
  @execution_manifest_file.ExecutionManifestFileLimits::new(
    max_manifest_bytes=ceilings.max_execution_manifest_bytes(),
    max_json_depth=RECIPE_JSON_DEPTH,
    max_modules=operation_count,
    max_operations=operation_count,
    device_memory=memory,
    launch_contract=contracts,
    artifact_file=artifact_files,
  ) catch {
    _ => raise AdmissionRejected(ExecutionLimits)
  }
}

///|
fn admit_execution_manifest(
  kernel_root : @approved_fs.ApprovedRoot,
  kernel_config : @kernels.KernelConfig,
  plan : @plan.ModelPlan,
  weights : @device_materialize.DeviceWeightFileInspection,
  layout : @device_layout.DeviceKvLayout,
  target : @catalog.DeviceTarget,
  step_limits : @device_step.DeviceStepLimits,
  limits : @execution_manifest_file.ExecutionManifestFileLimits,
  luna_approval : @worker_wire.WorkerBootstrapLunaApproval?,
  luna_verifier : @luna_capability_manifest.LunaStartupExternalApprovalHandoff?,
) -> @execution_manifest_file.PagedExecutionAdmission raise RuntimeDescriptorFileError {
  let digest = @execution_manifest_file.ExecutionManifestDigest::from_sha256(
    kernel_config.expected_manifest_digest().as_hex(),
  ) catch {
    _ => raise AdmissionRejected(ExecutionManifest)
  }
  @execution_manifest_file.load_paged(
    kernel_root,
    kernel_config.manifest_locator(),
    digest,
    plan,
    weights.layout(),
    layout,
    target,
    step_limits,
    limits,
    luna_approval~,
    luna_verifier~,
  ) catch {
    _ => raise AdmissionRejected(ExecutionManifest)
  }
}

///|
fn admit_bootstrap_limits(
  operation_count : Int,
) -> @device_step.DeviceBootstrapLimits raise RuntimeDescriptorFileError {
  @device_step.DeviceBootstrapLimits::new(
    max_modules=operation_count,
    max_entry_points=operation_count,
    max_steps=operation_count,
    max_operands_per_step=RECIPE_MAX_OPERANDS,
    max_symbol_bytes=RECIPE_MAX_SYMBOL_BYTES,
    max_canonical_bytes=RECIPE_MAX_CANONICAL_BYTES,
  ) catch {
    _ => raise AdmissionRejected(BootstrapManifest)
  }
}

///|
fn admit_bootstrap_manifest(
  execution : @execution_manifest_file.PagedExecutionAdmission,
  device_ordinal : Int,
  limits : @device_step.DeviceBootstrapLimits,
) -> @device_step.DeviceWorkerBootstrapManifest raise RuntimeDescriptorFileError {
  @device_step.admit_device_worker_bootstrap(
    execution.blueprint(),
    execution.artifacts(),
    luna_authorization=execution.luna_authorization(),
    device_ordinal~,
    limits,
  ) catch {
    _ => raise AdmissionRejected(BootstrapManifest)
  }
}

///|
fn require_bootstrap_digest(
  claims : RuntimeDescriptorKernelClaims,
  manifest : @device_step.DeviceWorkerBootstrapManifest,
) -> Unit raise RuntimeDescriptorFileError {
  let expected = @worker_wire.WorkerBootstrapDigest::from_sha256(
    claims.bootstrap_digest,
  ) catch {
    _ => raise AdmissionRejected(BootstrapDigest)
  }
  if manifest.digest() != expected {
    raise AdmissionRejected(BootstrapDigest)
  }
}
