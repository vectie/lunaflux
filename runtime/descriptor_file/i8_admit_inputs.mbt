///|
fn admit_i8_model(
  root : @approved_fs.ApprovedRoot,
  claims : I8RuntimeDescriptorModelClaims,
  ceilings : @worker_wire.WorkerBootstrapCeilings,
) -> @spec.LlamaModelMetadata raise RuntimeDescriptorFileError {
  let content = @spec.ContentDigest::from_sha256(claims.content_digest) catch {
    _ => raise AdmissionRejected(ModelIdentity)
  }
  let locator = @approved_fs.ApprovedRelativeLocator::new(claims.config_locator) catch {
    _ => raise AdmissionRejected(ModelConfig)
  }
  let digest = @config_file.ModelConfigDigest::from_sha256(claims.config_digest) catch {
    _ => raise AdmissionRejected(ModelConfig)
  }
  let limits = @config_file.ModelConfigFileLimits::new(
    maximum_bytes=ceilings
      .max_model_config_bytes()
      .to_int64()
      .min(RECIPE_MAX_CONFIG_BYTES),
    maximum_depth=RECIPE_JSON_DEPTH,
  ) catch {
    _ => raise AdmissionRejected(ModelConfig)
  }
  @config_file.load(root, locator, digest, content, limits) catch {
    _ => raise AdmissionRejected(ModelConfig)
  }
}

///|
fn admit_i8_plan(
  model : @spec.LlamaModelMetadata,
  max_batch_rows : Int,
) -> @plan.ModelPlan raise RuntimeDescriptorFileError {
  let batch = @llama.LlamaPagedBatchEnvelope::new(max_batch_rows) catch {
    _ => raise AdmissionRejected(ModelPlan)
  }
  @llama.build_paged_symmetric_i8_weight_only_v1(
    model,
    @llama.LlamaI8WeightOnlyPolicy::untied_v1(),
    batch~,
  ) catch {
    _ => raise AdmissionRejected(ModelPlan)
  }
}

///|
fn admit_i8_device_capability(
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
  let numeric = @device.admit_symmetric_i8_weight_only_per_output_channel_v1(
    capability,
  ) catch {
    _ => raise AdmissionRejected(DeviceNumericCapability)
  }
  let target = @catalog.DeviceTarget::new(
    compute_major=claims.compute_major,
    compute_minor=claims.compute_minor,
    supports_bf16=claims.supports_bf16,
    supports_cublas_lt=claims.supports_cublas_lt,
  ) catch {
    _ => raise AdmissionRejected(DeviceTarget)
  }
  (capability, numeric, target)
}

///|
fn inspect_i8_weights(
  root : @approved_fs.ApprovedRoot,
  plan : @plan.ModelPlan,
  claims : I8RuntimeDescriptorModelClaims,
  ceilings : @worker_wire.WorkerBootstrapCeilings,
) -> @numeric_weight_file.NumericWeightFileInspection raise RuntimeDescriptorFileError {
  let locator = @approved_fs.ApprovedRelativeLocator::new(
    claims.numeric_weights_locator,
  ) catch {
    _ => raise AdmissionRejected(NumericWeightInspection)
  }
  let digest = @numeric_weight_file.NumericWeightArtifactDigest::from_sha256(
    claims.numeric_weight_artifact_digest,
  ) catch {
    _ => raise AdmissionRejected(NumericWeightInspection)
  }
  let limits = @numeric_weight_file.NumericWeightFileLimits::new(
    max_file_bytes=ceilings.max_weight_file_bytes(),
    max_header_bytes=RECIPE_MAX_WEIGHT_HEADER_BYTES,
    transfer_chunk_bytes=RECIPE_TRANSFER_CHUNK_BYTES,
    max_tensor_count=plan.tensor_count(),
  ) catch {
    _ => raise AdmissionRejected(NumericWeightInspection)
  }
  @numeric_weight_file.inspect_file(
    root,
    plan,
    locator,
    expected_artifact_digest=digest,
    limits~,
    alignment_bytes=RECIPE_ALIGNMENT_BYTES,
    max_arena_bytes=ceilings.max_weight_arena_bytes(),
  ) catch {
    _ => raise AdmissionRejected(NumericWeightInspection)
  }
}

///|
fn admit_i8_policy_limits(
  plan : @plan.ModelPlan,
) -> @i8_inert_capability_admission.I8InertKernelCapabilityAdmissionLimits raise RuntimeDescriptorFileError {
  @i8_inert_capability_admission.I8InertKernelCapabilityAdmissionLimits::new(
    max_operations=plan.operations().length(),
    max_weights=plan.tensor_count(),
    max_canonical_bytes=RECIPE_MAX_CANONICAL_BYTES,
  ) catch {
    _ => raise AdmissionRejected(I8ProductionPolicy)
  }
}
