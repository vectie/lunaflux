///|
const INHERITED_INFERENCE_CREDENTIAL_MAX_BYTES : Int = 4096

///|
const INHERITED_INFERENCE_CREDENTIAL_HEADER_BYTES : Int = 12

///|
pub(all) enum InheritedInferenceCredentialRule {
  CredentialDescriptorUnavailable
  CredentialDescriptorInvalid
  CredentialFrameInvalid
  CredentialChannelLost
  CredentialStateInvalid
  CredentialPolicyInvalid
} derive(Debug, Eq)

///|
pub(all) suberror InheritedInferenceCredentialError {
  InheritedInferenceCredentialFailed(InheritedInferenceCredentialRule)
} derive(Debug, Eq)

///|
#valtype
pub(all) enum InheritedInferenceCredentialProgress {
  InheritedInferenceCredentialPending
  InheritedInferenceCredentialReady
  InheritedInferenceCredentialFailedProgress
} derive(Debug, Eq)

///|
pub struct InheritedInferenceCredentialOwner {
  priv native : @native.InferenceCredentialHandle
  priv header : FixedArray[Byte]
  priv credential : FixedArray[Byte]
  priv mut header_cursor : Int
  priv mut credential_cursor : Int
  priv mut credential_length : Int
  priv mut phase : Int
  priv mut failure_rule : Int
}

///|
/// Opaque startup authentication policy plus its non-observable source bound.
/// It has no debug, credential, length, digest, or storage getter and is
/// consumed exactly once by authenticated instance-policy admission.
pub struct InheritedInferenceAuthPolicy {
  priv policy : @api_auth.LunaApiAuthPolicy
  priv credential_length : Int
  priv maximum_credential_bytes : Int
  priv mut consumed : Bool
}
