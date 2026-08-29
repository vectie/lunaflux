///|
const I8_RUNTIME_SCHEMA : String = "lunaflux.runtime.i8.v2"

///|
const I8_RUNTIME_POLICY : String = "production_symmetric_i8_weight_only_per_output_channel_f32_scale_f32_accumulate_bf16_output_v1"

///|
fn parse_i8_descriptor(
  bytes : BytesView,
  limits : RuntimeDescriptorFileLimits,
) -> I8RuntimeDescriptorClaims raise RuntimeDescriptorFileError {
  if bytes.length().to_int64() > limits.max_descriptor_bytes() {
    raise FileTooLarge(maximum=limits.max_descriptor_bytes())
  }
  let text = @utf8.decode(bytes) catch { _ => raise InvalidUtf8 }
  @json_guard.reject_duplicate_object_keys(
    text,
    max_depth=limits.max_json_depth(),
  ) catch {
    InvalidDepthLimit => raise MalformedJson
    DepthLimitExceeded(_) =>
      raise JsonDepthExceeded(maximum=limits.max_json_depth())
    DuplicateObjectKey(depth~, index~) => raise DuplicateField(depth~, index~)
    MalformedStructure => raise MalformedJson
  }
  let value = @json.parse(text) catch { _ => raise MalformedJson }
  guard value is Object(root) else { raise MalformedJson }
  validate_fields(root, Root, [
    "schema_version", "model", "kernels", "execution", "worker_limits", "inference_limits",
    "ceilings",
  ])
  if require_string(root, "schema_version", Root) != I8_RUNTIME_SCHEMA {
    raise invalid_field(Root, UnsupportedValue)
  }
  {
    model: parse_i8_model(root),
    kernels: parse_i8_kernels(root),
    execution: parse_i8_execution(root),
    worker: parse_worker(root),
    inference: parse_inference(root),
    ceilings: parse_ceilings(root),
  }
}

///|
fn parse_i8_model(
  root : Map[String, Json],
) -> I8RuntimeDescriptorModelClaims raise RuntimeDescriptorFileError {
  let object = require_object(root, "model", Root, index=1)
  validate_fields(object, Model, [
    "config_locator", "config_sha256", "content_sha256", "numeric_weights_locator",
    "numeric_weight_artifact_sha256", "max_batch_rows",
  ])
  {
    config_locator: require_string(object, "config_locator", Model),
    config_digest: require_string(object, "config_sha256", Model, index=1),
    content_digest: require_string(object, "content_sha256", Model, index=2),
    numeric_weights_locator: require_string(
      object,
      "numeric_weights_locator",
      Model,
      index=3,
    ),
    numeric_weight_artifact_digest: require_string(
      object,
      "numeric_weight_artifact_sha256",
      Model,
      index=4,
    ),
    max_batch_rows: require_int(object, "max_batch_rows", Model, index=5),
  }
}

///|
fn parse_i8_kernels(
  root : Map[String, Json],
) -> I8RuntimeDescriptorKernelClaims raise RuntimeDescriptorFileError {
  let object = require_object(root, "kernels", Root, index=2)
  validate_fields(object, Kernels, [
    "manifest_locator", "manifest_sha256", "policy", "admitted_bootstrap_v3_sha256",
  ])
  let policy = require_string(object, "policy", Kernels, index=2)
  if policy != I8_RUNTIME_POLICY {
    raise invalid_field(Kernels, UnsupportedValue, index=2)
  }
  {
    manifest_locator: require_string(object, "manifest_locator", Kernels),
    manifest_digest: require_string(object, "manifest_sha256", Kernels, index=1),
    policy,
    bootstrap_v3_digest: require_string(
      object,
      "admitted_bootstrap_v3_sha256",
      Kernels,
      index=3,
    ),
  }
}

///|
fn parse_i8_execution(
  root : Map[String, Json],
) -> I8RuntimeDescriptorExecutionClaims raise RuntimeDescriptorFileError {
  let object = require_object(root, "execution", Root, index=3)
  validate_fields(object, Execution, [
    "device_ordinal", "compute_major", "compute_minor", "total_memory_bytes", "supports_bf16",
    "supports_cublas_lt", "tokens_per_page", "total_page_count", "model_generation",
  ])
  {
    device_ordinal: require_int(object, "device_ordinal", Execution),
    compute_major: require_int(object, "compute_major", Execution, index=1),
    compute_minor: require_int(object, "compute_minor", Execution, index=2),
    total_memory_bytes: require_int64(
      object,
      "total_memory_bytes",
      Execution,
      index=3,
    ),
    supports_bf16: require_bool(object, "supports_bf16", Execution, index=4),
    supports_cublas_lt: require_bool(
      object,
      "supports_cublas_lt",
      Execution,
      index=5,
    ),
    tokens_per_page: require_int(object, "tokens_per_page", Execution, index=6),
    total_page_count: require_int(
      object,
      "total_page_count",
      Execution,
      index=7,
    ),
    model_generation: require_uint64(
      object,
      "model_generation",
      Execution,
      index=8,
    ),
  }
}
