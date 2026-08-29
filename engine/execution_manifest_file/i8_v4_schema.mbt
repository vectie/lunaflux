///|
fn parse_v4_numeric_claim(
  root : Map[String, Json],
  model_plan : @plan.ModelPlan,
  device_numeric : @device.DeviceNumericCapability,
) -> ManifestV4NumericClaim raise ExecutionManifestFileError {
  let value = require_object(root, "numeric_authorization", Root)
  validate_fields(value, NumericAuthorization, [
    "numeric_schema_sha256", "i8_admission_sha256", "device_numeric_sha256", "weight_artifact_sha256",
  ])
  let numeric_schema_digest = require_string(
    value,
    "numeric_schema_sha256",
    NumericAuthorization,
  )
  let i8_admission_digest = require_string(
    value,
    "i8_admission_sha256",
    NumericAuthorization,
  )
  let device_numeric_digest = require_string(
    value,
    "device_numeric_sha256",
    NumericAuthorization,
  )
  let weight_artifact_digest = @numeric_weight_file.NumericWeightArtifactDigest::from_sha256(
    require_string(value, "weight_artifact_sha256", NumericAuthorization),
  ) catch {
    _ => raise invalid_field(NumericAuthorization, InvalidSha256)
  }
  if !is_lower_sha256(numeric_schema_digest) ||
    !is_lower_sha256(i8_admission_digest) ||
    !is_lower_sha256(device_numeric_digest) {
    raise invalid_field(NumericAuthorization, InvalidSha256)
  }
  if numeric_schema_digest != model_plan.numeric_binding().digest().as_hex() {
    raise IdentityMismatch(NumericSchemaDigest)
  }
  if device_numeric_digest != device_numeric.digest().as_hex() {
    raise IdentityMismatch(DeviceNumericDigest)
  }
  { i8_admission_digest, weight_artifact_digest }
}

///|
fn parse_manifest_v4(
  bytes : BytesView,
  model_plan : @plan.ModelPlan,
  device_numeric : @device.DeviceNumericCapability,
  target : @catalog.DeviceTarget,
  limits : ExecutionManifestFileLimits,
) -> ManifestV4Claims raise ExecutionManifestFileError {
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
    "schema_version", "model_identity", "numeric_authorization", "device_target",
    "catalog_version", "profile", "modules", "operations",
  ])
  if require_nonnegative_int(root, "schema_version", Root) != 4 {
    raise invalid_field(Root, UnsupportedValue)
  }
  if require_nonnegative_int(root, "catalog_version", Root) !=
    @catalog.CatalogVersion::v4().as_int() {
    raise IdentityMismatch(CatalogVersion)
  }
  validate_identity_claim(root, model_plan.identity())
  validate_target_claim(root, target)
  let numeric = parse_v4_numeric_claim(root, model_plan, device_numeric)
  let profile_id = parse_profile_claim(root)
  let modules = parse_module_claims(root, limits)
  let operations = parse_operation_claims(root, modules, model_plan, limits)
  { profile_id, modules, operations, numeric }
}
