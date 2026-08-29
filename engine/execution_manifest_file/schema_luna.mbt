///|
fn require_luna_sha256(
  object : Map[String, Json],
  field : String,
  index : Int,
) -> String raise ExecutionManifestFileError {
  let value = require_string(object, field, LunaAuthorization, index~)
  if !is_lower_sha256(value) {
    raise invalid_field(LunaAuthorization, InvalidSha256, index~)
  }
  value
}

///|
fn parse_luna_authorization_claim(
  root : Map[String, Json],
  model_plan : @plan.ModelPlan,
) -> ManifestLunaAuthorizationClaim raise ExecutionManifestFileError {
  let object = require_object(root, "luna_authorization", LunaAuthorization)
  validate_fields(object, LunaAuthorization, [
    "operation_id", "specialization_locator", "specialization_sha256", "capability_locator",
    "capability_sha256", "signature_algorithm", "key_identity_sha256", "signature_sha256",
  ])
  let operation_id = require_nonnegative_int(
    object,
    "operation_id",
    LunaAuthorization,
  )
  guard model_plan.operations().get(operation_id) is Some(operation) else {
    raise invalid_field(LunaAuthorization, UnsupportedValue)
  }
  if operation.id().as_int() != operation_id ||
    operation.kind() != ResidualAdd ||
    operation.required_capability() != @plan.KernelCapabilityId::residual_add() {
    raise invalid_field(LunaAuthorization, UnsupportedValue)
  }
  let signature_algorithm = require_string(
    object,
    "signature_algorithm",
    LunaAuthorization,
    index=5,
  )
  if signature_algorithm != "external_ed25519_v1" {
    raise invalid_field(LunaAuthorization, UnsupportedValue, index=5)
  }
  {
    operation_id,
    specialization_locator: require_string(
      object,
      "specialization_locator",
      LunaAuthorization,
      index=1,
    ),
    specialization_digest: require_luna_sha256(
      object, "specialization_sha256", 2,
    ),
    capability_locator: require_string(
      object,
      "capability_locator",
      LunaAuthorization,
      index=3,
    ),
    capability_digest: require_luna_sha256(object, "capability_sha256", 4),
    key_identity: require_luna_sha256(object, "key_identity_sha256", 6),
    signature_digest: require_luna_sha256(object, "signature_sha256", 7),
  }
}
