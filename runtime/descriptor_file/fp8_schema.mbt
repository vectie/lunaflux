///|
const FP8_RUNTIME_SCHEMA : String = "lunaflux.runtime.fp8-reusable.v3"

///|
const FP8_RUNTIME_POLICY : String = "externally_approved_finite_e4m3_w8a8_reusable_paged_v3"

///|
fn parse_fp8_descriptor(
  bytes : BytesView,
  limits : RuntimeDescriptorFileLimits,
) -> Fp8RuntimeDescriptorClaims raise RuntimeDescriptorFileError {
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
  if require_string(root, "schema_version", Root) != FP8_RUNTIME_SCHEMA {
    raise invalid_field(Root, UnsupportedValue)
  }
  let kernels = require_object(root, "kernels", Root, index=2)
  validate_fields(kernels, Kernels, [
    "manifest_locator", "manifest_sha256", "policy",
  ])
  let policy = require_string(kernels, "policy", Kernels, index=2)
  if policy != FP8_RUNTIME_POLICY {
    raise invalid_field(Kernels, UnsupportedValue, index=2)
  }
  {
    model: parse_i8_model(root),
    kernels: {
      manifest_locator: require_string(kernels, "manifest_locator", Kernels),
      manifest_digest: require_string(
        kernels,
        "manifest_sha256",
        Kernels,
        index=1,
      ),
      policy,
    },
    execution: parse_i8_execution(root),
    worker: parse_worker(root),
    inference: parse_inference(root),
    ceilings: parse_ceilings(root),
  }
}
