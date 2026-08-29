///|
fn admit_fp8_device_capability(
  claims : I8RuntimeDescriptorExecutionClaims,
) -> (
  @device.DeviceCapability,
  @device.DeviceNumericCapability,
  @catalog.DeviceTarget,
) raise RuntimeDescriptorFileError {
  if !((claims.compute_major == 8 && claims.compute_minor == 9) ||
    (claims.compute_major == 9 && claims.compute_minor == 0) ||
    (claims.compute_major == 12 && claims.compute_minor == 0)) ||
    !claims.supports_bf16 ||
    !claims.supports_cublas_lt {
    raise AdmissionRejected(DeviceCapability)
  }
  let capability = @device.DeviceCapability::from_observed_facts(
    compute_major=claims.compute_major,
    compute_minor=claims.compute_minor,
    total_memory_bytes=claims.total_memory_bytes,
    supports_bf16=claims.supports_bf16,
    supports_cublas_lt=claims.supports_cublas_lt,
  ) catch {
    _ => raise AdmissionRejected(DeviceCapability)
  }
  let numeric = @device.admit_finite_fp8_e4m3_w8a8(capability) catch {
    _ => raise AdmissionRejected(DeviceNumericCapability)
  }
  let target = @catalog.DeviceTarget::from_capability(capability) catch {
    _ => raise AdmissionRejected(DeviceTarget)
  }
  (capability, numeric, target)
}

///|
pub fn load_fp8_reusable_v3(
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  model_root_label : StringView,
  kernel_root_label : StringView,
  descriptor_locator : @approved_fs.ApprovedRelativeLocator,
  expected_digest : RuntimeDescriptorDigest,
  limits : RuntimeDescriptorFileLimits,
  approval : @worker_wire.WorkerBootstrapLunaApproval,
  verifier : @luna_capability_manifest.LunaStartupExternalApprovalHandoff,
) -> Fp8RuntimeDescriptorAdmission raise RuntimeDescriptorFileError {
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
  load_fp8_reusable_v3_bound(
    model_root,
    kernel_root,
    model_root_label,
    kernel_root_label,
    descriptor_locator,
    expected_digest,
    limits,
    approval,
    verifier=Some(verifier),
  )
}

///|
/// Shared exact reconstruction after source-root/target-label authority joins.
fn load_fp8_reusable_v3_bound(
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  model_root_label : StringView,
  kernel_root_label : StringView,
  descriptor_locator : @approved_fs.ApprovedRelativeLocator,
  expected_digest : RuntimeDescriptorDigest,
  limits : RuntimeDescriptorFileLimits,
  approval : @worker_wire.WorkerBootstrapLunaApproval,
  verifier? : @luna_capability_manifest.LunaStartupExternalApprovalHandoff? = None,
) -> Fp8RuntimeDescriptorAdmission raise RuntimeDescriptorFileError {
  let bytes = read_descriptor_snapshot(
    model_root, descriptor_locator, expected_digest, limits,
  )
  let claims = parse_fp8_descriptor(bytes, limits)
  if claims.kernels.policy != FP8_RUNTIME_POLICY ||
    claims.model.max_batch_rows != claims.worker.max_plan_rows {
    raise AdmissionRejected(DeviceCapability)
  }
  let (_, numeric, target) = admit_fp8_device_capability(claims.execution)
  let ceilings = admit_ceilings(claims.ceilings)
  let worker = admit_worker_limits(claims.worker)
  let inference = admit_inference_limits(claims.inference)
  let model = admit_i8_model(model_root, claims.model, ceilings)
  let batch = @llama.LlamaPagedBatchEnvelope::new(claims.model.max_batch_rows) catch {
    _ => raise AdmissionRejected(ModelPlan)
  }
  let plan = @llama.build_paged_finite_e4m3_fp8_w8a8_v2(
    model,
    @llama.LlamaFp8W8A8V2Policy::untied_v2(),
    batch~,
  ) catch {
    _ => raise AdmissionRejected(ModelPlan)
  }
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
  let weight_locator = @approved_fs.ApprovedRelativeLocator::new(
    claims.model.numeric_weights_locator,
  ) catch {
    _ => raise AdmissionRejected(NumericWeightInspection)
  }
  let weight_digest = @numeric_weight_file.NumericWeightArtifactDigest::from_sha256(
    claims.model.numeric_weight_artifact_digest,
  ) catch {
    _ => raise AdmissionRejected(NumericWeightInspection)
  }
  let weight_limits = @numeric_weight_file.NumericWeightFileLimits::new(
    max_file_bytes=ceilings.max_weight_file_bytes(),
    max_header_bytes=RECIPE_MAX_WEIGHT_HEADER_BYTES,
    transfer_chunk_bytes=RECIPE_TRANSFER_CHUNK_BYTES,
    max_tensor_count=plan.tensor_count(),
  ) catch {
    _ => raise AdmissionRejected(NumericWeightInspection)
  }
  let inspection = @llama_fp8_weights.inspect_paged_file_v2(
    model_root,
    model,
    @llama.LlamaFp8W8A8V2Policy::untied_v2(),
    weight_locator,
    expected_artifact_digest=weight_digest,
    limits=weight_limits,
    alignment_bytes=RECIPE_ALIGNMENT_BYTES,
    max_arena_bytes=ceilings.max_weight_arena_bytes(),
    batch~,
  ) catch {
    _ => raise AdmissionRejected(NumericWeightInspection)
  }
  let step_limits = admit_step_limits(worker)
  let operation_count = plan.operations().length()
  let execution_limits = admit_execution_limits(operation_count, ceilings)
  guard verifier is Some(verifier) else {
    raise AdmissionRejected(ExecutionManifest)
  }
  let execution = @execution_manifest_file.load_paged_fp8_v5(
    kernel_root,
    @approved_fs.ApprovedRelativeLocator::new(claims.kernels.manifest_locator) catch {
      _ => raise AdmissionRejected(ExecutionManifest)
    },
    @execution_manifest_file.ExecutionManifestDigest::from_sha256(
      claims.kernels.manifest_digest,
    ) catch {
      _ => raise AdmissionRejected(ExecutionManifest)
    },
    plan,
    inspection,
    kv_layout,
    numeric,
    target,
    step_limits,
    execution_limits,
    @numeric_capability_manifest.Fp8KernelManifestLimits::new(
      max_entries=operation_count,
      max_canonical_bytes=RECIPE_MAX_CANONICAL_BYTES,
    ) catch {
      _ => raise AdmissionRejected(ExecutionManifest)
    },
    @fp8_startup_admission.Fp8StartupAdmissionLimits::new(
      max_canonical_bytes=RECIPE_MAX_CANONICAL_BYTES,
    ) catch {
      _ => raise AdmissionRejected(ExecutionManifest)
    },
    @fp8_runtime_recipe.Fp8RuntimeRecipeV2Limits::new(
      max_launches=operation_count,
      max_workspace_arena_bytes=ceilings.max_activation_arena_bytes(),
      max_canonical_bytes=RECIPE_MAX_CANONICAL_BYTES,
    ) catch {
      _ => raise AdmissionRejected(ExecutionManifest)
    },
    @fp8_runtime_recipe.Fp8ReusablePagedRecipeV3Limits::new(
      max_launches=operation_count,
      max_workspace_arena_bytes=ceilings.max_activation_arena_bytes(),
      max_canonical_bytes=RECIPE_MAX_CANONICAL_BYTES,
    ) catch {
      _ => raise AdmissionRejected(ExecutionManifest)
    },
    approval,
    @fp8_release_authority.Fp8ReleaseAuthorityLimits::new(
      max_manifest_bytes=ceilings.max_execution_manifest_bytes(),
      max_module_bytes=ceilings.max_module_bytes(),
      max_total_module_bytes=ceilings.max_total_module_bytes(),
      max_operations=operation_count,
    ) catch {
      _ => raise AdmissionRejected(ExecutionManifest)
    },
    verifier=Some(verifier),
  ) catch {
    _ => raise AdmissionRejected(ExecutionManifest)
  }
  let source = @worker_wire.admit_fp8_reusable_bootstrap_source_v9(
    @worker_wire.WorkerBootstrapRoots::new(
      model_root=model_root_label,
      kernel_root=kernel_root_label,
    ),
    @worker_wire.WorkerFp8BootstrapModelSource::new(
      claims.model.config_locator,
      claims.model.numeric_weights_locator,
      claims.model.config_digest,
      claims.model.numeric_weight_artifact_digest,
    ),
    @worker_wire.WorkerBootstrapKernelSource::new(
      manifest_locator=claims.kernels.manifest_locator,
      manifest_digest_sha256=claims.kernels.manifest_digest,
      luna_approval=Some(approval),
    ),
    @worker_wire.WorkerFp8BootstrapExecution::new(
      claims.execution.compute_major,
      claims.execution.compute_minor,
      claims.execution.total_memory_bytes,
      claims.execution.supports_bf16,
      claims.execution.supports_cublas_lt,
      claims.execution.tokens_per_page,
      claims.execution.total_page_count,
      execution.reusable_recipe().digest().as_hex(),
    ),
    ceilings,
  ) catch {
    _ => raise AdmissionRejected(BootstrapSource)
  }
  let startup = @worker_wire.WorkerStartupContract::new(
    identity=plan.identity(),
    bootstrap_digest=@worker_wire.WorkerBootstrapDigest::from_sha256(
      execution.release_authority().digest().as_hex(),
    ),
    bootstrap_source_digest=source.digest(),
    device_ordinal=claims.execution.device_ordinal,
    model_generation_value=claims.execution.model_generation,
    worker_limits=worker,
    inference_limits=inference,
    has_predecessor=false,
    predecessor_value=0UL,
  ) catch {
    _ => raise AdmissionRejected(StartupContract)
  }
  {
    descriptor_digest: expected_digest,
    target,
    device_ordinal: claims.execution.device_ordinal,
    execution,
    source,
    startup,
  }
}
