///|
fn parse_mistral_descriptor(
  bytes : BytesView,
  limits : RuntimeDescriptorFileLimits,
) -> MistralRuntimeDescriptorClaims raise RuntimeDescriptorFileError {
  if bytes.length().to_int64() > limits.max_descriptor_bytes() {
    raise FileTooLarge(maximum=limits.max_descriptor_bytes())
  }
  let text = @utf8.decode(bytes) catch { _ => raise InvalidUtf8 }
  @json_guard.reject_duplicate_object_keys(
    text,
    max_depth=limits.max_json_depth(),
  ) catch {
    DepthLimitExceeded(_) =>
      raise JsonDepthExceeded(maximum=limits.max_json_depth())
    DuplicateObjectKey(depth~, index~) => raise DuplicateField(depth~, index~)
    _ => raise MalformedJson
  }
  let value = @json.parse(text) catch { _ => raise MalformedJson }
  guard value is Object(root) else { raise MalformedJson }
  validate_fields(root, Root, [
    "schema_version", "model", "kernels", "execution", "worker_limits", "inference_limits",
    "ceilings",
  ])
  let schema_version = require_string(root, "schema_version", Root)
  if schema_version != "lunaflux.runtime.mistral_bf16.v1" &&
    schema_version != "lunaflux.runtime.mistral_bf16.v2" {
    raise invalid_field(Root, UnsupportedValue)
  }
  let object = require_object(root, "model", Root, index=1)
  validate_fields(object, Model, [
    "family", "config_locator", "config_sha256", "content_sha256", "numeric_weights_locator",
    "numeric_weight_artifact_sha256", "weight_manifest_sha256", "sliding_window_tokens",
    "max_batch_rows",
  ])
  let kernels = parse_kernels(root)
  let has_extensions = kernels.fused_runtime is Some(_) ||
    kernels.sampling_runtime == EmbeddedCudaGreedyV1
  if (schema_version == "lunaflux.runtime.mistral_bf16.v2") != has_extensions {
    raise invalid_field(Kernels, UnsupportedValue)
  }
  {
    model: {
      family: require_string(object, "family", Model),
      config_locator: require_string(object, "config_locator", Model, index=1),
      config_digest: require_string(object, "config_sha256", Model, index=2),
      content_digest: require_string(object, "content_sha256", Model, index=3),
      numeric_weights_locator: require_string(
        object,
        "numeric_weights_locator",
        Model,
        index=4,
      ),
      numeric_weight_artifact_digest: require_string(
        object,
        "numeric_weight_artifact_sha256",
        Model,
        index=5,
      ),
      weight_manifest_digest: require_string(
        object,
        "weight_manifest_sha256",
        Model,
        index=6,
      ),
      sliding_window_tokens: require_int(
        object,
        "sliding_window_tokens",
        Model,
        index=7,
      ),
      max_batch_rows: require_int(object, "max_batch_rows", Model, index=8),
    },
    kernels,
    execution: parse_execution(root),
    worker: parse_worker(root),
    inference: parse_inference(root),
    ceilings: parse_ceilings(root, graph_memory_required=true),
  }
}
