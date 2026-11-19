///|
const INHERITED_INFERENCE_CREDENTIAL_MAX_BYTES : Int32 = 4096

///|
const INHERITED_INFERENCE_CREDENTIAL_HEADER_BYTES : Int32 = 12

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
  priv mut header_cursor : Int32
  priv mut credential_cursor : Int32
  priv mut credential_length : Int32
  priv mut phase : Int32
  priv mut failure_rule : Int32
}

///|
/// Opaque startup authentication policy plus its non-observable source bound.
/// It has no debug, credential, length, digest, or storage getter and is
/// consumed exactly once by authenticated instance-policy admission.
pub struct InheritedInferenceAuthPolicy {
  priv policy : @api_auth.LunaApiAuthPolicy
  priv credential_length : Int32
  priv maximum_credential_bytes : Int32
  priv mut consumed : Bool
}

///|
pub extend InheritedInferenceCredentialRule with @debug.Debug::{to_repr}

///|
pub extend InheritedInferenceCredentialRule with Eq::{not_equal, equal}

///|
pub extend InheritedInferenceCredentialError with @debug.Debug::{to_repr}

///|
pub extend InheritedInferenceCredentialError with Eq::{not_equal, equal}

///|
pub extend InheritedInferenceCredentialProgress with @debug.Debug::{to_repr}

///|
pub extend InheritedInferenceCredentialProgress with Eq::{not_equal, equal}
