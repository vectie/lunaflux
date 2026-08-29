///|
fn parse_v5_numeric_claim(
  root : Map[String, Json],
  plan : @plan.ModelPlan,
  numeric : @device.DeviceNumericCapability,
) -> ManifestV5NumericClaim raise ExecutionManifestFileError {
  let value = require_object(root, "numeric_authorization", Root)
  validate_fields(value, NumericAuthorization, [
    "numeric_schema_sha256", "device_numeric_sha256", "weight_artifact_sha256", "fp8_kernel_manifest_sha256",
    "fp8_runtime_recipe_v2_sha256", "fp8_reusable_recipe_v3_sha256", "workspace_storage_sha256",
  ])
  let schema = require_string(
    value,
    "numeric_schema_sha256",
    NumericAuthorization,
  )
  let device = require_string(
    value,
    "device_numeric_sha256",
    NumericAuthorization,
  )
  let kernel = require_string(
    value,
    "fp8_kernel_manifest_sha256",
    NumericAuthorization,
  )
  let exact = require_string(
    value,
    "fp8_runtime_recipe_v2_sha256",
    NumericAuthorization,
  )
  let reusable = require_string(
    value,
    "fp8_reusable_recipe_v3_sha256",
    NumericAuthorization,
  )
  let workspace = require_string(
    value,
    "workspace_storage_sha256",
    NumericAuthorization,
  )
  if schema != plan.numeric_binding().digest().as_hex() ||
    device != numeric.digest().as_hex() {
    raise IdentityMismatch(NumericSchemaDigest)
  }
  for digest in [kernel, exact, reusable, workspace] {
    if !is_lower_sha256(digest) {
      raise invalid_field(NumericAuthorization, InvalidSha256)
    }
  }
  {
    weight_artifact_digest: @numeric_weight_file.NumericWeightArtifactDigest::from_sha256(
      require_string(value, "weight_artifact_sha256", NumericAuthorization),
    ) catch {
      _ => raise invalid_field(NumericAuthorization, InvalidSha256)
    },
    kernel_manifest_digest: kernel,
    runtime_recipe_v2_digest: exact,
    reusable_recipe_v3_digest: reusable,
    workspace_storage_digest: workspace,
  }
}

///|
fn parse_manifest_v5(
  bytes : BytesView,
  plan : @plan.ModelPlan,
  numeric : @device.DeviceNumericCapability,
  target : @catalog.DeviceTarget,
  limits : ExecutionManifestFileLimits,
) -> ManifestV5Claims raise ExecutionManifestFileError {
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
    "catalog_version", "profile", "modules", "operations", "fp8_release",
  ])
  if require_nonnegative_int(root, "schema_version", Root) != 5 ||
    require_nonnegative_int(root, "catalog_version", Root) !=
    @catalog.CatalogVersion::v4().as_int() {
    raise invalid_field(Root, UnsupportedValue)
  }
  validate_identity_claim(root, plan.identity())
  validate_target_claim(root, target)
  let modules = parse_module_claims(root, limits)
  let release_object = require_object(root, "fp8_release", Root)
  validate_fields(release_object, NumericAuthorization, [
    "manifest_locator", "operations",
  ])
  let release_values = require_array(
    release_object,
    "operations",
    NumericAuthorization,
  )
  if release_values.is_empty() ||
    release_values.length() > limits.max_operations() {
    raise invalid_field(NumericAuthorization, CountExceeded)
  }
  let release_operations : Array[ManifestV5ReleaseOperationClaim] = []
  for index, value in release_values {
    guard value is Object(operation) else {
      raise invalid_field(NumericAuthorization, ExpectedObject, index~)
    }
    validate_fields(operation, NumericAuthorization, [
      "operation_id", "module_locator", "source_digest_sha256", "recipe_digest_sha256",
    ])
    let source = require_string(
      operation,
      "source_digest_sha256",
      NumericAuthorization,
      index~,
    )
    let recipe = require_string(
      operation,
      "recipe_digest_sha256",
      NumericAuthorization,
      index~,
    )
    if !is_lower_sha256(source) || !is_lower_sha256(recipe) {
      raise invalid_field(NumericAuthorization, InvalidSha256, index~)
    }
    release_operations.push({
      operation: @plan.OperationId::new(
        require_nonnegative_int(
          operation,
          "operation_id",
          NumericAuthorization,
          index~,
        ),
      ) catch {
        _ => raise invalid_field(NumericAuthorization, UnsupportedValue, index~)
      },
      module_locator: require_string(
        operation,
        "module_locator",
        NumericAuthorization,
        index~,
      ),
      source_digest: source,
      recipe_digest: recipe,
    })
  }
  {
    profile_id: parse_profile_claim(root),
    modules,
    operations: parse_operation_claims(root, modules, plan, limits),
    numeric: parse_v5_numeric_claim(root, plan, numeric),
    release: {
      manifest_locator: require_string(
        release_object,
        "manifest_locator",
        NumericAuthorization,
      ),
      operations: release_operations,
    },
  }
}
