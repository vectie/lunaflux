///|
/// Admits a complete inert single-device dense-Llama runtime descriptor.
///
/// `expected_digest` is an independent deployment input; the descriptor cannot
/// authenticate itself. Model and kernel authorities remain distinct even
/// when their canonical labels name the same directory. Each label is bound to
/// its exact caller-owned pinned root before any descendant is opened. The
/// caller retains both roots and must close each one on every outcome.
pub fn load(
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  model_root_label : StringView,
  kernel_root_label : StringView,
  descriptor_locator : @approved_fs.ApprovedRelativeLocator,
  expected_digest : RuntimeDescriptorDigest,
  limits : RuntimeDescriptorFileLimits,
  luna_approval? : @worker_wire.WorkerBootstrapLunaApproval? = None,
  luna_verifier? : @luna_capability_manifest.LunaStartupExternalApprovalHandoff? = None,
) -> RuntimeDescriptorAdmission raise RuntimeDescriptorFileError {
  bind_root_label(
    model_root,
    model_root_label,
    BindModelRoot,
    limits.max_descriptor_bytes(),
  )
  bind_root_label(
    kernel_root,
    kernel_root_label,
    BindKernelRoot,
    limits.max_descriptor_bytes(),
  )
  load_bound(
    model_root,
    kernel_root,
    model_root_label,
    kernel_root_label,
    descriptor_locator,
    expected_digest,
    limits,
    luna_approval~,
    luna_verifier~,
  )
}

///|
fn load_bound(
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  model_root_label : StringView,
  kernel_root_label : StringView,
  descriptor_locator : @approved_fs.ApprovedRelativeLocator,
  expected_digest : RuntimeDescriptorDigest,
  limits : RuntimeDescriptorFileLimits,
  luna_approval? : @worker_wire.WorkerBootstrapLunaApproval? = None,
  luna_verifier? : @luna_capability_manifest.LunaStartupExternalApprovalHandoff? = None,
) -> RuntimeDescriptorAdmission raise RuntimeDescriptorFileError {
  let bytes = read_descriptor_snapshot(
    model_root, descriptor_locator, expected_digest, limits,
  )
  let claims = parse_descriptor(bytes, limits)
  let graph_capture_ceiling = admit_graph_capture_ceiling(claims.ceilings)
  let ceilings = admit_ceilings(claims.ceilings)
  let kernel_config = admit_kernel_config(claims.kernels)
  let model = admit_model(model_root, claims.model, ceilings)
  let plan = admit_plan(model, claims.model.max_batch_rows)
  let worker = admit_worker_limits(claims.worker)
  let inference = admit_inference_limits(claims.inference)
  let target = admit_target(claims.execution)
  let layout = admit_kv_layout(plan, claims.execution, ceilings)
  let weights = admit_weight_inspection(
    model_root,
    model,
    plan,
    claims.model,
    ceilings,
  )
  let step_limits = admit_step_limits(worker)
  let operation_count = plan.operations().length()
  let execution_limits = admit_execution_limits(operation_count, ceilings)
  let execution = admit_execution_manifest(
    kernel_root, kernel_config, plan, weights, layout, target, step_limits, execution_limits,
    luna_approval, luna_verifier,
  )
  let graph_memory = @runtime_resolved.resolve_graph_memory(
    execution.blueprint(),
    graph_capture_ceiling,
    weight_reserved_bytes=weights.layout().arena_bytes(),
    activation_workspace_reserved_bytes=execution.memory_plan().arena_bytes(),
    kv_reserved_bytes=layout.arena_bytes(),
  ) catch {
    _ => raise AdmissionRejected(GraphMemory)
  }
  let bootstrap_limits = admit_bootstrap_limits(operation_count)
  let bootstrap = admit_bootstrap_manifest(
    execution,
    claims.execution.device_ordinal,
    bootstrap_limits,
  )
  require_bootstrap_digest(claims.kernels, bootstrap)
  let source = admit_source(
    model_root_label, kernel_root_label, claims, ceilings, luna_approval,
  )
  let startup = admit_startup_contract(
    bootstrap,
    source,
    claims.execution.model_generation,
    worker,
    inference,
  )
  let sampling_policy = @device_worker.admit_bootstrap_sampling_policy(
    source, plan, execution,
  ) catch {
    _ => raise AdmissionRejected(DeviceWorkerPlan)
  }
  let worker_plan = @device_worker.admit_plan(
    startup~,
    model~,
    weight_inspection=weights,
    execution~,
    bootstrap_limits~,
    sampling_policy~,
  ) catch {
    _ => raise AdmissionRejected(DeviceWorkerPlan)
  }
  {
    descriptor_digest: expected_digest,
    target,
    device_ordinal: claims.execution.device_ordinal,
    execution,
    source,
    startup,
    _worker_plan: worker_plan,
    graph_memory,
  }
}
