///|
fn parse_manifest(
  bytes : BytesView,
  model_plan : @plan.ModelPlan,
  target : @catalog.DeviceTarget,
  limits : ExecutionManifestFileLimits,
) -> ManifestClaims raise ExecutionManifestFileError {
  if bytes.length().to_int64() > limits.max_manifest_bytes() {
    raise FileTooLarge(maximum=limits.max_manifest_bytes())
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
    "schema_version", "model_identity", "device_target", "catalog_version", "profile",
    "modules", "operations",
  ])
  if require_nonnegative_int(root, "schema_version", Root) != 1 {
    raise invalid_field(Root, UnsupportedValue)
  }
  if require_nonnegative_int(root, "catalog_version", Root) !=
    @catalog.CatalogVersion::v3().as_int() {
    raise IdentityMismatch(CatalogVersion)
  }
  validate_identity_claim(root, model_plan.identity())
  validate_target_claim(root, target)
  let profile_id = parse_profile_claim(root)
  let modules = parse_module_claims(root, limits)
  let operations = parse_operation_claims(root, modules, model_plan, limits)
  { profile_id, modules, operations }
}
