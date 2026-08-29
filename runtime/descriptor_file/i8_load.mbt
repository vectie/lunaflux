///|
/// Admits a strict, digest-pinned production-I8 runtime descriptor without
/// opening a device or granting allocation/readiness authority.
pub fn load_i8_v2(
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  model_root_label : StringView,
  kernel_root_label : StringView,
  descriptor_locator : @approved_fs.ApprovedRelativeLocator,
  expected_digest : RuntimeDescriptorDigest,
  limits : RuntimeDescriptorFileLimits,
) -> I8RuntimeDescriptorAdmission raise RuntimeDescriptorFileError {
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
  load_i8_v2_bound(
    model_root, kernel_root, model_root_label, kernel_root_label, descriptor_locator,
    expected_digest, limits,
  )
}

///|
fn load_i8_v2_bound(
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  model_root_label : StringView,
  kernel_root_label : StringView,
  descriptor_locator : @approved_fs.ApprovedRelativeLocator,
  expected_digest : RuntimeDescriptorDigest,
  limits : RuntimeDescriptorFileLimits,
) -> I8RuntimeDescriptorAdmission raise RuntimeDescriptorFileError {
  let bytes = read_descriptor_snapshot(
    model_root, descriptor_locator, expected_digest, limits,
  )
  let claims = parse_i8_descriptor(bytes, limits)
  preflight_i8_claims(claims)
  let ceilings = admit_ceilings(claims.ceilings)
  let worker = admit_worker_limits(claims.worker)
  let inference = admit_inference_limits(claims.inference)
  let (_, numeric, target) = admit_i8_device_capability(claims.execution)
  let model = admit_i8_model(model_root, claims.model, ceilings)
  let plan = admit_i8_plan(model, claims.model.max_batch_rows)
  let legacy_execution : RuntimeDescriptorExecutionClaims = {
    device_ordinal: claims.execution.device_ordinal,
    compute_major: claims.execution.compute_major,
    compute_minor: claims.execution.compute_minor,
    supports_bf16: claims.execution.supports_bf16,
    supports_cublas_lt: claims.execution.supports_cublas_lt,
    tokens_per_page: claims.execution.tokens_per_page,
    total_page_count: claims.execution.total_page_count,
    model_generation: claims.execution.model_generation,
  }
  let kv_layout = admit_kv_layout(plan, legacy_execution, ceilings)
  let weights = inspect_i8_weights(model_root, plan, claims.model, ceilings)
  let step_limits = admit_step_limits(worker)
  let operation_count = plan.operations().length()
  let execution_limits = admit_execution_limits(operation_count, ceilings)
  let bootstrap_limits = admit_bootstrap_limits(operation_count)
  let execution = load_i8_execution_v4(
    kernel_root, claims, plan, weights, kv_layout, numeric, target, step_limits,
    execution_limits, bootstrap_limits,
  )
  require_i8_bootstrap_v3_digest(claims.kernels, execution)
  let source = admit_i8_source_v6(
    model_root_label, kernel_root_label, claims, ceilings,
  )
  let startup = admit_i8_startup_contract(
    execution,
    source,
    claims.execution.model_generation,
    worker,
    inference,
  )
  {
    descriptor_digest: expected_digest,
    target,
    device_ordinal: claims.execution.device_ordinal,
    weights,
    execution,
    source,
    startup,
  }
}
