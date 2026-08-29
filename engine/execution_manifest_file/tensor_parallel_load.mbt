///|
/// Loads and admits one exact rank-local tensor-parallel execution manifest.
///
/// The canonical bounded manifest snapshot, parser, semantic catalog builder,
/// and artifact-file loader are shared with ordinary manifest admission. The
/// full model KV layout is derived only as inert semantic catalog evidence;
/// launch contracts and the physical execution plan use the authenticated
/// rank-local layout. This function never constructs a full weight layout and
/// never opens a device or backend resource.
pub fn load_tensor_parallel(
  approved_root : @approved_fs.ApprovedRoot,
  locator : @approved_fs.ApprovedRelativeLocator,
  expected_digest : ExecutionManifestDigest,
  model_plan : @plan.ModelPlan,
  model_generation : @worker_protocol.ModelPlanGeneration,
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan,
  sharded_inspection : @device_materialize.DeviceShardedWeightFileInspection,
  device_plan : @tensor_parallel_device_plan.TensorParallelDevicePlan,
  local_kv_layout : @device_layout.DeviceKvLayout,
  local_kv_binding : @tensor_parallel_kv_plan.TensorParallelKvRankBinding,
  target : @catalog.DeviceTarget,
  device_step_limits : @device_step.DeviceStepLimits,
  limits : ExecutionManifestFileLimits,
  launch_limits : @tensor_parallel_launch_contract.TensorParallelLaunchContractLimits,
  execution_limits : @tensor_parallel_execution_plan.TensorParallelExecutionPlanLimits,
  external_approval? : @worker_wire.WorkerBootstrapLunaApproval? = None,
) -> TensorParallelExecutionAdmission raise ExecutionManifestFileError {
  let full_kv_layout = validate_tensor_parallel_envelope(
    model_plan, rank_plan, sharded_inspection, device_plan, local_kv_layout, local_kv_binding,
    model_generation, target, device_step_limits, limits,
  )
  let bytes = read_manifest_snapshot(
    approved_root, locator, expected_digest, limits,
  )
  let claims = parse_manifest(bytes, model_plan, target, limits)
  match external_approval {
    None => ()
    Some(approval) =>
      if approval.approved_source() != expected_digest.as_hex() {
        raise AdmissionRejected(LunaExternalApproval)
      }
  }
  if claims.luna_authorization is Some(_) {
    raise AdmissionRejected(TensorParallelSpecialization)
  }
  let resolved = derive_catalog(claims, model_plan, full_kv_layout, target)
  let profile = derive_tensor_parallel_profile(claims, device_step_limits)
  let contracts = derive_tensor_parallel_contracts(
    claims, model_plan, rank_plan, local_kv_layout, resolved, target, profile, launch_limits,
  )
  let artifact_source = derive_artifact_source(claims, limits.artifact_file())
  let artifacts = @artifact_file.load_tensor_parallel_admitted(
    approved_root,
    artifact_source,
    contracts,
    limits.artifact_file(),
  ) catch {
    _ => raise AdmissionRejected(ArtifactFiles)
  }
  let execution_plan = @tensor_parallel_execution_plan.admit(
    model_plan, contracts, device_plan, artifacts, profile, execution_limits,
  ) catch {
    _ => raise AdmissionRejected(TensorParallelExecutionPlan)
  }
  {
    model_plan,
    model_generation,
    rank_plan,
    device_plan,
    local_kv_layout,
    local_kv_binding,
    profile,
    contracts,
    artifacts,
    execution_plan,
  }
}
