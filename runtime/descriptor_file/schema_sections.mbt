///|
fn parse_model(
  root : Map[String, Json],
) -> RuntimeDescriptorModelClaims raise RuntimeDescriptorFileError {
  let object = require_object(root, "model", Root, index=1)
  validate_fields(object, Model, [
    "config_locator", "config_sha256", "weights_locator", "content_sha256", "max_batch_rows",
  ])
  {
    config_locator: require_string(object, "config_locator", Model),
    config_digest: require_string(object, "config_sha256", Model, index=1),
    weights_locator: require_string(object, "weights_locator", Model, index=2),
    content_digest: require_string(object, "content_sha256", Model, index=3),
    max_batch_rows: require_int(object, "max_batch_rows", Model, index=4),
  }
}

///|
fn parse_kernels(
  root : Map[String, Json],
) -> RuntimeDescriptorKernelClaims raise RuntimeDescriptorFileError {
  let object = require_object(root, "kernels", Root, index=2)
  validate_required_and_optional_fields(
    object,
    Kernels,
    [
      "manifest_locator", "manifest_sha256", "policy", "admitted_bootstrap_sha256",
    ],
    ["fused_runtime", "sampling_runtime"],
  )
  let fused_runtime = match object.get("fused_runtime") {
    None => None
    Some(Object(fused)) => {
      validate_fields(fused, Kernels, ["locator", "sha256"])
      Some({
        locator: require_string(fused, "locator", Kernels, index=4),
        digest: require_string(fused, "sha256", Kernels, index=5),
      })
    }
    Some(_) => raise invalid_field(Kernels, ExpectedObject, index=4)
  }
  let sampling_runtime = match object.get("sampling_runtime") {
    None => @worker_wire.HostSamplingRuntime
    Some(String("embedded_cuda_greedy_v1")) => EmbeddedCudaGreedyV1
    Some(_) => raise invalid_field(Kernels, UnsupportedValue, index=6)
  }
  {
    manifest_locator: require_string(object, "manifest_locator", Kernels),
    manifest_digest: require_string(object, "manifest_sha256", Kernels, index=1),
    policy: require_string(object, "policy", Kernels, index=2),
    bootstrap_digest: require_string(
      object,
      "admitted_bootstrap_sha256",
      Kernels,
      index=3,
    ),
    fused_runtime,
    sampling_runtime,
  }
}

///|
fn parse_execution(
  root : Map[String, Json],
) -> RuntimeDescriptorExecutionClaims raise RuntimeDescriptorFileError {
  let object = require_object(root, "execution", Root, index=3)
  validate_fields(object, Execution, [
    "device_ordinal", "compute_major", "compute_minor", "supports_bf16", "supports_cublas_lt",
    "tokens_per_page", "total_page_count", "model_generation",
  ])
  {
    device_ordinal: require_int(object, "device_ordinal", Execution),
    compute_major: require_int(object, "compute_major", Execution, index=1),
    compute_minor: require_int(object, "compute_minor", Execution, index=2),
    supports_bf16: require_bool(object, "supports_bf16", Execution, index=3),
    supports_cublas_lt: require_bool(
      object,
      "supports_cublas_lt",
      Execution,
      index=4,
    ),
    tokens_per_page: require_int(object, "tokens_per_page", Execution, index=5),
    total_page_count: require_int(
      object,
      "total_page_count",
      Execution,
      index=6,
    ),
    model_generation: require_uint64(
      object,
      "model_generation",
      Execution,
      index=7,
    ),
  }
}

///|
fn parse_worker(
  root : Map[String, Json],
) -> RuntimeDescriptorWorkerClaims raise RuntimeDescriptorFileError {
  let object = require_object(root, "worker_limits", Root, index=4)
  validate_fields(object, WorkerLimits, [
    "max_prefill_rows", "max_decode_rows", "max_plan_rows", "max_plan_tokens", "max_plan_pages",
    "max_capabilities", "max_completion_slots", "max_sequence_tokens", "max_token_id",
  ])
  {
    max_prefill_rows: require_int(object, "max_prefill_rows", WorkerLimits),
    max_decode_rows: require_int(
      object,
      "max_decode_rows",
      WorkerLimits,
      index=1,
    ),
    max_plan_rows: require_int(object, "max_plan_rows", WorkerLimits, index=2),
    max_plan_tokens: require_int(
      object,
      "max_plan_tokens",
      WorkerLimits,
      index=3,
    ),
    max_plan_pages: require_int(object, "max_plan_pages", WorkerLimits, index=4),
    max_capabilities: require_int(
      object,
      "max_capabilities",
      WorkerLimits,
      index=5,
    ),
    max_completion_slots: require_int(
      object,
      "max_completion_slots",
      WorkerLimits,
      index=6,
    ),
    max_sequence_tokens: require_int(
      object,
      "max_sequence_tokens",
      WorkerLimits,
      index=7,
    ),
    max_token_id: require_int(object, "max_token_id", WorkerLimits, index=8),
  }
}

///|
fn parse_inference(
  root : Map[String, Json],
) -> RuntimeDescriptorInferenceClaims raise RuntimeDescriptorFileError {
  let object = require_object(root, "inference_limits", Root, index=5)
  validate_fields(object, InferenceLimits, [
    "max_text_bytes", "max_input_tokens", "max_new_tokens", "max_context_tokens",
    "max_token_id", "max_stop_token_ids", "max_stop_strings", "max_stop_string_bytes",
    "max_trace_bytes", "max_cache_scope_bytes", "max_decoded_delta_bytes", "max_deadline_millis",
    "max_top_k", "max_temperature",
  ])
  {
    max_text_bytes: require_int(object, "max_text_bytes", InferenceLimits),
    max_input_tokens: require_int(
      object,
      "max_input_tokens",
      InferenceLimits,
      index=1,
    ),
    max_new_tokens: require_int(
      object,
      "max_new_tokens",
      InferenceLimits,
      index=2,
    ),
    max_context_tokens: require_int(
      object,
      "max_context_tokens",
      InferenceLimits,
      index=3,
    ),
    max_token_id: require_int(object, "max_token_id", InferenceLimits, index=4),
    max_stop_token_ids: require_int(
      object,
      "max_stop_token_ids",
      InferenceLimits,
      index=5,
    ),
    max_stop_strings: require_int(
      object,
      "max_stop_strings",
      InferenceLimits,
      index=6,
    ),
    max_stop_string_bytes: require_int(
      object,
      "max_stop_string_bytes",
      InferenceLimits,
      index=7,
    ),
    max_trace_bytes: require_int(
      object,
      "max_trace_bytes",
      InferenceLimits,
      index=8,
    ),
    max_cache_scope_bytes: require_int(
      object,
      "max_cache_scope_bytes",
      InferenceLimits,
      index=9,
    ),
    max_decoded_delta_bytes: require_int(
      object,
      "max_decoded_delta_bytes",
      InferenceLimits,
      index=10,
    ),
    max_deadline_millis: require_uint64(
      object,
      "max_deadline_millis",
      InferenceLimits,
      index=11,
    ),
    max_top_k: require_int(object, "max_top_k", InferenceLimits, index=12),
    max_temperature: require_number(
      object,
      "max_temperature",
      InferenceLimits,
      index=13,
    ),
  }
}

///|
fn parse_ceilings(
  root : Map[String, Json],
  graph_memory_required? : Bool = false,
) -> RuntimeDescriptorCeilingClaims raise RuntimeDescriptorFileError {
  let object = require_object(root, "ceilings", Root, index=6)
  if graph_memory_required {
    validate_fields(object, Ceilings, [
      "max_model_config_bytes", "max_weight_file_bytes", "max_weight_arena_bytes",
      "max_activation_arena_bytes", "max_kv_arena_bytes", "max_execution_manifest_bytes",
      "max_module_bytes", "max_total_module_bytes", "max_graph_capture_bytes",
    ])
  } else {
    validate_fields(object, Ceilings, [
      "max_model_config_bytes", "max_weight_file_bytes", "max_weight_arena_bytes",
      "max_activation_arena_bytes", "max_kv_arena_bytes", "max_execution_manifest_bytes",
      "max_module_bytes", "max_total_module_bytes",
    ])
  }
  {
    max_model_config_bytes: require_int(
      object,
      "max_model_config_bytes",
      Ceilings,
    ),
    max_weight_file_bytes: require_int64(
      object,
      "max_weight_file_bytes",
      Ceilings,
      index=1,
    ),
    max_weight_arena_bytes: require_int64(
      object,
      "max_weight_arena_bytes",
      Ceilings,
      index=2,
    ),
    max_activation_arena_bytes: require_int64(
      object,
      "max_activation_arena_bytes",
      Ceilings,
      index=3,
    ),
    max_kv_arena_bytes: require_int64(
      object,
      "max_kv_arena_bytes",
      Ceilings,
      index=4,
    ),
    max_execution_manifest_bytes: require_int64(
      object,
      "max_execution_manifest_bytes",
      Ceilings,
      index=5,
    ),
    max_module_bytes: require_int64(
      object,
      "max_module_bytes",
      Ceilings,
      index=6,
    ),
    max_total_module_bytes: require_int64(
      object,
      "max_total_module_bytes",
      Ceilings,
      index=7,
    ),
    max_graph_capture_bytes: if graph_memory_required {
      Some(require_int64(object, "max_graph_capture_bytes", Ceilings, index=8))
    } else {
      None
    },
  }
}
