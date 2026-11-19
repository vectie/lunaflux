///|
pub extend DeepSeekV4QuantizedPayloadEncoding with @debug.Debug::{to_repr}

///|
pub extend DeepSeekV4QuantizedPayloadEncoding with Eq::{not_equal, equal}

///|
pub extend DeepSeekV4QuantizedInterpretationUnavailable with @debug.Debug::{
  to_repr,
}

///|
pub extend DeepSeekV4QuantizedInterpretationUnavailable with Eq::{
  not_equal,
  equal,
}

///|
pub extend DeepSeekV4ScalarInterpretationAvailability with @debug.Debug::{
  to_repr,
}

///|
pub extend DeepSeekV4ScalarInterpretationAvailability with Eq::{
  not_equal,
  equal,
}

///|
pub extend DeepSeekV4QuantizedPayloadError with @debug.Debug::{to_repr}

///|
pub extend DeepSeekV4QuantizedPayloadError with Eq::{not_equal, equal}

///|
pub extend DeepSeekV4QuantizedPayloadLimits with @debug.Debug::{to_repr}

///|
pub extend DeepSeekV4QuantizedPayloadLimits with Eq::{not_equal, equal}

///|
pub extend DeepSeekV4QuantizedBlockShape with @debug.Debug::{to_repr}

///|
pub extend DeepSeekV4QuantizedBlockShape with Eq::{not_equal, equal}

///|
pub extend DeepSeekV4QuantizedPayloadContractDigest with @debug.Debug::{to_repr}

///|
pub extend DeepSeekV4QuantizedPayloadContractDigest with Eq::{not_equal, equal}

///|
pub extend DeepSeekV4QuantizedPayloadContractDigest with Hash::{
  hash,
  hash_combine,
}
