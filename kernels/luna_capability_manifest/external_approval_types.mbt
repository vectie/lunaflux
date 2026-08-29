///|
const LUNA_EXTERNAL_APPROVAL_POLICY_SCHEMA : String = "lunaflux.external-approval-verifier-policy.v1"

///|
const LUNA_EXTERNAL_SIGNED_APPROVAL_SCHEMA : String = "lunaflux.external-signed-approval.v1"

///|
const LUNA_EXTERNAL_APPROVAL_HMAC_DOMAIN : String = "lunaflux.external-approval-hmac-sha256.v1"

///|
pub struct LunaExternalApprovalPolicy {
  priv rollout_digest : LunaCapabilityManifestDigest
  priv verifier_key_id : String
  priv authority_id : String
}

///|
fn luna_external_approval_identifier(value : StringView) -> Bool {
  if value.is_empty() || value.length() > 128 {
    return false
  }
  for unit in value {
    if !(unit is ('A'..='Z' | 'a'..='z' | '0'..='9' | '_' | '-' | '.')) {
      return false
    }
  }
  true
}

///|
pub fn LunaExternalApprovalPolicy::new(
  schema~ : String,
  rollout_digest~ : LunaCapabilityManifestDigest,
  verifier_key_id~ : String,
  authority_id~ : String,
) -> LunaExternalApprovalPolicy raise LunaCapabilityManifestError {
  if schema != LUNA_EXTERNAL_APPROVAL_POLICY_SCHEMA ||
    !luna_external_approval_identifier(verifier_key_id) ||
    !luna_external_approval_identifier(authority_id) {
    raise Invalid(ExternalApproval)
  }
  { rollout_digest, verifier_key_id, authority_id }
}

///|
pub struct LunaExternalSignedApproval {
  priv schema : String
  priv rollout_digest : LunaCapabilityManifestDigest
  priv verifier_key_id : String
  priv authority_id : String
  priv manifest_digest : LunaCapabilityManifestDigest
  priv algorithm : LunaDetachedSignatureAlgorithm
  priv key_identity : LunaSignatureKeyIdentityDigest
  priv signature_digest : LunaDetachedSignatureDigest
  priv approved_source : LunaApprovedImmutableSourceIdentity
  priv authentication_tag : LunaDetachedSignatureDigest
}

///|
/// Constructs inert signed evidence. This validates only its bounded schema;
/// it grants no approval, manifest, artifact, device, or runtime authority.
pub fn LunaExternalSignedApproval::new(
  schema~ : String,
  rollout_digest~ : LunaCapabilityManifestDigest,
  verifier_key_id~ : String,
  authority_id~ : String,
  manifest_digest~ : LunaCapabilityManifestDigest,
  algorithm~ : LunaDetachedSignatureAlgorithm,
  key_identity~ : LunaSignatureKeyIdentityDigest,
  signature_digest~ : LunaDetachedSignatureDigest,
  approved_source~ : LunaApprovedImmutableSourceIdentity,
  authentication_tag~ : LunaDetachedSignatureDigest,
) -> LunaExternalSignedApproval raise LunaCapabilityManifestError {
  if schema != LUNA_EXTERNAL_SIGNED_APPROVAL_SCHEMA ||
    !luna_external_approval_identifier(verifier_key_id) ||
    !luna_external_approval_identifier(authority_id) {
    raise Invalid(ExternalApproval)
  }
  {
    schema,
    rollout_digest,
    verifier_key_id,
    authority_id,
    manifest_digest,
    algorithm,
    key_identity,
    signature_digest,
    approved_source,
    authentication_tag,
  }
}

///|
/// Startup-owned verifier state. There is deliberately no public arbitrary-key
/// constructor. Only the fixed inherited startup capability can construct it.
pub struct LunaExternalApprovalVerifier {
  priv policy : LunaExternalApprovalPolicy
  priv key : FixedArray[Byte]
  priv mut closed : Bool
}

///|
#valtype
pub(all) enum LunaStartupVerifierKeyProgress {
  LunaStartupVerifierKeyPending
  LunaStartupVerifierKeyReady
  LunaStartupVerifierKeyFailedProgress
} derive(Debug, Eq)

///|
pub struct LunaStartupVerifierKeyOwner {
  priv native : @promotion_key.PromotionVerifierKeyHandle
  priv header : FixedArray[Byte]
  priv payload : FixedArray[Byte]
  priv mut header_cursor : Int
  priv mut payload_cursor : Int
  priv mut payload_length : Int
  priv mut key_length : Int
  priv mut verifier_id_length : Int
  priv mut authority_id_length : Int
  priv mut phase : Int
}

///|
/// One exact startup approval capability. It exposes neither key bytes nor
/// policy identifiers and can authenticate at most one manifest claim.
pub struct LunaStartupExternalApprovalHandoff {
  priv verifier : LunaExternalApprovalVerifier
  priv rollout_digest : LunaCapabilityManifestDigest
  priv verifier_key_id : String
  priv authority_id : String
  priv authentication_tag : LunaDetachedSignatureDigest
  priv mut consumed : Bool
}

///|
/// One decoded attestation from the root-bound parent's private startup
/// channel. It contains no deployment verifier key and is consumable once.
pub struct LunaParentStartupApprovalWitness {
  priv manifest_digest : LunaCapabilityManifestDigest
  priv key_identity : LunaSignatureKeyIdentityDigest
  priv signature_digest : LunaDetachedSignatureDigest
  priv approved_source : LunaApprovedImmutableSourceIdentity
  priv bootstrap_digest : LunaCapabilityManifestDigest
  priv source_digest : LunaCapabilityManifestDigest
  priv model_generation : UInt64
  priv device_ordinal : Int
  priv mut consumed : Bool
}
