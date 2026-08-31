///|
fn parse_descriptor(
  bytes : BytesView,
  limits : RuntimeDescriptorFileLimits,
) -> RuntimeDescriptorClaims raise RuntimeDescriptorFileError {
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
  let schema_version = require_string(root, "schema_version", Root)
  let graph_memory_required = match schema_version {
    "lunaflux.runtime.v1" | "lunaflux.runtime.v5" => false
    "lunaflux.runtime.v2" | "lunaflux.runtime.v3" | "lunaflux.runtime.v4" =>
      true
    _ => raise invalid_field(Root, UnsupportedValue)
  }
  let kernels = parse_kernels(root)
  let has_extensions = kernels.fused_runtime is Some(_) ||
    kernels.sampling_runtime == EmbeddedCudaGreedyV1
  let extension_schema = schema_version == "lunaflux.runtime.v4" ||
    schema_version == "lunaflux.runtime.v5"
  if extension_schema != has_extensions {
    raise invalid_field(Kernels, UnsupportedValue)
  }
  {
    model: parse_model(root),
    kernels,
    execution: parse_execution(root),
    worker: parse_worker(root),
    inference: parse_inference(root),
    ceilings: parse_ceilings(root, graph_memory_required~),
  }
}
