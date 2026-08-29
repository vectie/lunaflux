///|
const RECIPE_MAX_CONFIG_BYTES : Int64 = 16777216L

///|
const RECIPE_MAX_WEIGHT_HEADER_BYTES : Int = 16777216

///|
const RECIPE_JSON_DEPTH : Int = 16

///|
const RECIPE_MAX_SYMBOL_BYTES : Int = 1024

///|
const RECIPE_MAX_RANK : Int = 2

///|
const RECIPE_MAX_TENSOR_NAME_BYTES : Int = 256

///|
const RECIPE_TRANSFER_CHUNK_BYTES : Int = 4194304

///|
const RECIPE_ALIGNMENT_BYTES : Int64 = 256L

///|
const RECIPE_MAX_OPERANDS : Int = 32

///|
const RECIPE_MAX_CANONICAL_BYTES : Int = 16777216

///|
fn admit_kernel_config(
  claims : RuntimeDescriptorKernelClaims,
) -> @kernels.KernelConfig raise RuntimeDescriptorFileError {
  if claims.policy != "deployment_approved_aot_only" {
    raise AdmissionRejected(KernelConfig)
  }
  @kernels.new(
    claims.manifest_locator,
    claims.manifest_digest,
    DeploymentApprovedAotOnly,
  ) catch {
    _ => raise AdmissionRejected(KernelConfig)
  }
}

///|
fn admit_model(
  root : @approved_fs.ApprovedRoot,
  claims : RuntimeDescriptorModelClaims,
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
  let maximum = ceilings
    .max_model_config_bytes()
    .to_int64()
    .min(RECIPE_MAX_CONFIG_BYTES)
  let limits = @config_file.ModelConfigFileLimits::new(
    maximum_bytes=maximum,
    maximum_depth=RECIPE_JSON_DEPTH,
  ) catch {
    _ => raise AdmissionRejected(ModelConfig)
  }
  @config_file.load(root, locator, digest, content, limits) catch {
    _ => raise AdmissionRejected(ModelConfig)
  }
}

///|
fn admit_plan(
  model : @spec.LlamaModelMetadata,
  max_batch_rows : Int,
) -> @plan.ModelPlan raise RuntimeDescriptorFileError {
  let batch = @llama.LlamaPagedBatchEnvelope::new(max_batch_rows) catch {
    _ => raise AdmissionRejected(ModelPlan)
  }
  @llama.build_paged(model, batch~) catch {
    _ => raise AdmissionRejected(ModelPlan)
  }
}

///|
fn admit_worker_limits(
  claims : RuntimeDescriptorWorkerClaims,
) -> @worker_protocol.WorkerProtocolLimits raise RuntimeDescriptorFileError {
  @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=claims.max_prefill_rows,
    max_decode_rows=claims.max_decode_rows,
    max_plan_rows=claims.max_plan_rows,
    max_plan_tokens=claims.max_plan_tokens,
    max_plan_pages=claims.max_plan_pages,
    max_capabilities=claims.max_capabilities,
    max_completion_slots=claims.max_completion_slots,
    max_sequence_tokens=claims.max_sequence_tokens,
    max_token_id=claims.max_token_id,
  ) catch {
    _ => raise AdmissionRejected(WorkerLimits)
  }
}

///|
fn admit_inference_limits(
  claims : RuntimeDescriptorInferenceClaims,
) -> @inference.InferenceLimits raise RuntimeDescriptorFileError {
  @inference.InferenceLimits::new(
    max_text_bytes=claims.max_text_bytes,
    max_input_tokens=claims.max_input_tokens,
    max_new_tokens=claims.max_new_tokens,
    max_context_tokens=claims.max_context_tokens,
    max_token_id=claims.max_token_id,
    max_stop_token_ids=claims.max_stop_token_ids,
    max_stop_strings=claims.max_stop_strings,
    max_stop_string_bytes=claims.max_stop_string_bytes,
    max_trace_bytes=claims.max_trace_bytes,
    max_cache_scope_bytes=claims.max_cache_scope_bytes,
    max_decoded_delta_bytes=claims.max_decoded_delta_bytes,
    max_deadline_millis=claims.max_deadline_millis,
    max_top_k=claims.max_top_k,
    max_temperature=claims.max_temperature,
  ) catch {
    _ => raise AdmissionRejected(InferenceLimits)
  }
}

///|
fn admit_ceilings(
  claims : RuntimeDescriptorCeilingClaims,
) -> @worker_wire.WorkerBootstrapCeilings raise RuntimeDescriptorFileError {
  @worker_wire.WorkerBootstrapCeilings::new(
    max_model_config_bytes=claims.max_model_config_bytes,
    max_weight_file_bytes=claims.max_weight_file_bytes,
    max_weight_arena_bytes=claims.max_weight_arena_bytes,
    max_activation_arena_bytes=claims.max_activation_arena_bytes,
    max_kv_arena_bytes=claims.max_kv_arena_bytes,
    max_execution_manifest_bytes=claims.max_execution_manifest_bytes,
    max_module_bytes=claims.max_module_bytes,
    max_total_module_bytes=claims.max_total_module_bytes,
  ) catch {
    _ => raise AdmissionRejected(ExecutionLimits)
  }
}

///|
fn admit_graph_capture_ceiling(
  claims : RuntimeDescriptorCeilingClaims,
) -> Int64? raise RuntimeDescriptorFileError {
  match claims.max_graph_capture_bytes {
    None => None
    Some(value) => {
      if value <= 0L {
        raise AdmissionRejected(GraphMemory)
      }
      Some(value)
    }
  }
}

///|
fn admit_target(
  claims : RuntimeDescriptorExecutionClaims,
) -> @catalog.DeviceTarget raise RuntimeDescriptorFileError {
  if !claims.supports_bf16 {
    raise AdmissionRejected(DeviceTarget)
  }
  @catalog.DeviceTarget::new(
    compute_major=claims.compute_major,
    compute_minor=claims.compute_minor,
    supports_bf16=claims.supports_bf16,
    supports_cublas_lt=claims.supports_cublas_lt,
  ) catch {
    _ => raise AdmissionRejected(DeviceTarget)
  }
}

///|
fn admit_weight_inspection(
  root : @approved_fs.ApprovedRoot,
  model : @spec.LlamaModelMetadata,
  plan : @plan.ModelPlan,
  claims : RuntimeDescriptorModelClaims,
  ceilings : @worker_wire.WorkerBootstrapCeilings,
) -> @device_materialize.DeviceWeightFileInspection raise RuntimeDescriptorFileError {
  let exact_count = plan.operations().length()
  let safetensors = @safetensors.HeaderLimits::new(
    max_header_bytes=RECIPE_MAX_WEIGHT_HEADER_BYTES,
    max_tensor_count=exact_count,
    max_rank=RECIPE_MAX_RANK,
    max_name_length=RECIPE_MAX_TENSOR_NAME_BYTES,
    max_payload_bytes=ceilings.max_weight_file_bytes(),
    max_materialized_tensor_bytes=ceilings
      .max_weight_file_bytes()
      .min(ceilings.max_weight_arena_bytes()),
  ) catch {
    _ => raise AdmissionRejected(WeightInspection)
  }
  let limits = @device_materialize.DeviceWeightFileLimits::new(
    max_file_bytes=ceilings.max_weight_file_bytes(),
    transfer_chunk_bytes=RECIPE_TRANSFER_CHUNK_BYTES,
    safetensors~,
  ) catch {
    _ => raise AdmissionRejected(WeightInspection)
  }
  let locator = @approved_fs.ApprovedRelativeLocator::new(
    claims.weights_locator,
  ) catch {
    _ => raise AdmissionRejected(WeightInspection)
  }
  @device_materialize.inspect_file(
    root,
    model,
    plan,
    locator,
    limits~,
    alignment_bytes=RECIPE_ALIGNMENT_BYTES,
    max_arena_bytes=ceilings.max_weight_arena_bytes(),
  ) catch {
    _ => raise AdmissionRejected(WeightInspection)
  }
}
