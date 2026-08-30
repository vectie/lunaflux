///|
fn load_qwen3_config_snapshot(
  root : @approved_fs.ApprovedRoot,
  claims : Qwen3RuntimeDescriptorModelClaims,
  maximum : Int,
) -> @qwen3_spec.Qwen3ModelMetadata raise RuntimeDescriptorFileError {
  if claims.family != "qwen3" || !claims.tied_embeddings {
    raise AdmissionRejected(ModelConfig)
  }
  let content = @spec.ContentDigest::from_sha256(claims.content_digest) catch {
    _ => raise AdmissionRejected(ModelIdentity)
  }
  let file = root.open_file(
    @approved_fs.ApprovedRelativeLocator::new(claims.config_locator) catch {
      _ => raise AdmissionRejected(ModelConfig)
    },
  ) catch {
    _ => raise AdmissionRejected(ModelConfig)
  }
  let body : Result[Bytes, RuntimeDescriptorFileError] = Ok(
    file.read_immutable_snapshot(maximum_bytes=maximum.to_int64()) catch {
      _ => raise AdmissionRejected(ModelConfig)
    },
  ) catch {
    error => Err(error)
  }
  let close : Result[Unit, RuntimeDescriptorFileError] = Ok(
    file.close() catch {
      _ => raise AdmissionRejected(ModelConfig)
    },
  ) catch {
    error => Err(error)
  }
  let bytes = match body {
    Ok(value) => value
    Err(error) => raise error
  }
  match close {
    Err(error) => raise error
    Ok(_) => ()
  }
  if @crypto.bytes_to_hex_string(@crypto.sha256(bytes)) != claims.config_digest {
    raise AdmissionRejected(ModelConfig)
  }
  let metadata = @config_reader.parse_qwen3_bytes(
    bytes,
    content,
    max_bytes=maximum,
    max_depth=RECIPE_JSON_DEPTH,
  ) catch {
    _ => raise AdmissionRejected(ModelConfig)
  }
  if !metadata.spec().tied_embeddings() {
    raise AdmissionRejected(ModelConfig)
  }
  metadata
}

///|
fn inspect_qwen3_numeric_weights(
  root : @approved_fs.ApprovedRoot,
  metadata : @qwen3_spec.Qwen3ModelMetadata,
  batch : @qwen3.Qwen3PagedBatchEnvelope,
  claims : Qwen3RuntimeDescriptorModelClaims,
  ceilings : @worker_wire.WorkerBootstrapCeilings,
  tensor_count : Int,
) -> @qwen3_weights.Qwen3WeightFileInspection raise RuntimeDescriptorFileError {
  let inspection = @qwen3_weights.inspect_paged_file(
    root,
    metadata,
    @approved_fs.ApprovedRelativeLocator::new(claims.numeric_weights_locator) catch {
      _ => raise AdmissionRejected(NumericWeightInspection)
    },
    expected_artifact_digest=@numeric_weight_file.NumericWeightArtifactDigest::from_sha256(
      claims.numeric_weight_artifact_digest,
    ) catch {
      _ => raise AdmissionRejected(WeightArtifactDigest)
    },
    limits=@numeric_weight_file.NumericWeightFileLimits::new(
      max_file_bytes=ceilings.max_weight_file_bytes(),
      max_header_bytes=RECIPE_MAX_WEIGHT_HEADER_BYTES,
      transfer_chunk_bytes=RECIPE_TRANSFER_CHUNK_BYTES,
      max_tensor_count=tensor_count,
    ) catch {
      _ => raise AdmissionRejected(NumericWeightInspection)
    },
    alignment_bytes=RECIPE_ALIGNMENT_BYTES,
    max_arena_bytes=ceilings.max_weight_arena_bytes(),
    expected_route_manifest_sha256=claims.weight_manifest_digest,
    batch~,
  ) catch {
    _ => raise AdmissionRejected(NumericWeightInspection)
  }
  if inspection.route_manifest_sha256() != claims.weight_manifest_digest {
    raise AdmissionRejected(WeightArtifactDigest)
  }
  inspection
}

///|
fn admit_qwen3_source(
  model_root_label : StringView,
  kernel_root_label : StringView,
  claims : Qwen3RuntimeDescriptorClaims,
  ceilings : @worker_wire.WorkerBootstrapCeilings,
) -> @worker_wire.EncodedBootstrapSource raise RuntimeDescriptorFileError {
  let roots = @worker_wire.WorkerBootstrapRoots::new(
    model_root=model_root_label,
    kernel_root=kernel_root_label,
  ) catch {
    _ => raise AdmissionRejected(BootstrapSource)
  }
  let model = @worker_wire.WorkerQwen3BootstrapModelSource::new(
    config_locator=claims.model.config_locator,
    numeric_weights_locator=claims.model.numeric_weights_locator,
    config_digest_sha256=claims.model.config_digest,
    numeric_weight_artifact_sha256=claims.model.numeric_weight_artifact_digest,
    weight_manifest_sha256=claims.model.weight_manifest_digest,
    tied_embeddings=claims.model.tied_embeddings,
  ) catch {
    _ => raise AdmissionRejected(BootstrapSource)
  }
  let kernel = @worker_wire.WorkerBootstrapKernelSource::new(
    manifest_locator=claims.kernels.manifest_locator,
    manifest_digest_sha256=claims.kernels.manifest_digest,
    sampling_runtime=claims.kernels.sampling_runtime,
  ) catch {
    _ => raise AdmissionRejected(BootstrapSource)
  }
  let execution = @worker_wire.WorkerBootstrapExecution::new(
    compute_major=claims.execution.compute_major,
    compute_minor=claims.execution.compute_minor,
    supports_bf16=claims.execution.supports_bf16,
    supports_cublas_lt=claims.execution.supports_cublas_lt,
    tokens_per_page=claims.execution.tokens_per_page,
    total_page_count=claims.execution.total_page_count,
  ) catch {
    _ => raise AdmissionRejected(BootstrapSource)
  }
  @worker_wire.admit_qwen3_bf16_paged_bootstrap_source_v12(
    roots, model, kernel, execution, ceilings,
  ) catch {
    _ => raise AdmissionRejected(BootstrapSource)
  }
}

///|
fn load_qwen3_bound(
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  model_root_label : StringView,
  kernel_root_label : StringView,
  descriptor_locator : @approved_fs.ApprovedRelativeLocator,
  expected_digest : RuntimeDescriptorDigest,
  limits : RuntimeDescriptorFileLimits,
) -> RuntimeDescriptorAdmission raise RuntimeDescriptorFileError {
  let claims = parse_qwen3_descriptor(
    read_descriptor_snapshot(
      model_root, descriptor_locator, expected_digest, limits,
    ),
    limits,
  )
  let graph_capture_ceiling = admit_graph_capture_ceiling(claims.ceilings)
  let ceilings = admit_ceilings(claims.ceilings)
  let kernel_config = admit_kernel_config(claims.kernels)
  let metadata = load_qwen3_config_snapshot(
    model_root,
    claims.model,
    ceilings.max_model_config_bytes(),
  )
  let batch = @qwen3.Qwen3PagedBatchEnvelope::new(claims.model.max_batch_rows) catch {
    _ => raise AdmissionRejected(ModelPlan)
  }
  let plan = @qwen3.build_paged(metadata, batch~) catch {
    _ => raise AdmissionRejected(ModelPlan)
  }
  if plan.tensor_count() != metadata.spec().layer_count() * 11 + 2 ||
    plan.operations().length() != metadata.spec().layer_count() * 10 + 3 {
    raise AdmissionRejected(ModelPlan)
  }
  let worker = admit_worker_limits(claims.worker)
  let inference = admit_inference_limits(claims.inference)
  let target = admit_target(claims.execution)
  let layout = admit_kv_layout(plan, claims.execution, ceilings)
  let weights = inspect_qwen3_numeric_weights(
    model_root,
    metadata,
    batch,
    claims.model,
    ceilings,
    plan.tensor_count(),
  )
  let step_limits = admit_step_limits(worker)
  let execution_limits = admit_execution_limits(
    plan.operations().length(),
    ceilings,
  )
  let manifest_digest = @execution_manifest_file.ExecutionManifestDigest::from_sha256(
    kernel_config.expected_manifest_digest().as_hex(),
  ) catch {
    _ => raise AdmissionRejected(ExecutionManifest)
  }
  let execution = @execution_manifest_file.load_paged(
    kernel_root,
    kernel_config.manifest_locator(),
    manifest_digest,
    plan,
    weights.layout(),
    layout,
    target,
    step_limits,
    execution_limits,
  ) catch {
    _ => raise AdmissionRejected(ExecutionManifest)
  }
  let graph_memory = @runtime_resolved.resolve_graph_memory(
    execution.blueprint(),
    graph_capture_ceiling,
    weight_reserved_bytes=weights.layout().arena_bytes(),
    activation_workspace_reserved_bytes=execution.memory_plan().arena_bytes(),
    kv_reserved_bytes=layout.arena_bytes(),
  ) catch {
    _ => raise AdmissionRejected(GraphMemory)
  }
  let bootstrap_limits = admit_bootstrap_limits(plan.operations().length())
  let bootstrap = admit_bootstrap_manifest(
    execution,
    claims.execution.device_ordinal,
    bootstrap_limits,
  )
  require_bootstrap_digest(claims.kernels, bootstrap)
  let source = admit_qwen3_source(
    model_root_label, kernel_root_label, claims, ceilings,
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
  let worker_plan = @device_worker.admit_numeric_bf16_plan_v1(
    startup~,
    model_plan=plan,
    weight_inspection=weights.generic_inspection(),
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

///|
/// Admits the strict dense-Qwen3 BF16 v1 descriptor without probing another
/// model family or runtime recipe.
pub fn load_qwen3_v1(
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  model_root_label : StringView,
  kernel_root_label : StringView,
  descriptor_locator : @approved_fs.ApprovedRelativeLocator,
  expected_digest : RuntimeDescriptorDigest,
  limits : RuntimeDescriptorFileLimits,
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
  load_qwen3_bound(
    model_root, kernel_root, model_root_label, kernel_root_label, descriptor_locator,
    expected_digest, limits,
  )
}
