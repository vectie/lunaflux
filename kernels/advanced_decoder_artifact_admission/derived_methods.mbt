///|
pub extend AdvancedDecoderArtifactAdmissionError with @debug.Debug::{to_repr}

///|
pub extend AdvancedDecoderArtifactAdmissionError with Eq::{not_equal, equal}

///|
pub extend AdvancedDecoderArtifactAdmissionLimits with @debug.Debug::{to_repr}

///|
pub extend AdvancedDecoderArtifactAdmissionLimits with Eq::{not_equal, equal}

///|
pub extend AdvancedDecoderDeclaredQualificationDigest with @debug.Debug::{
  to_repr,
}

///|
pub extend AdvancedDecoderDeclaredQualificationDigest with Eq::{
  not_equal,
  equal,
}

///|
pub extend AdvancedDecoderDeclaredQualificationDigest with Hash::{
  hash,
  hash_combine,
}

///|
pub extend AdvancedDecoderLaunchOperandRole with @debug.Debug::{to_repr}

///|
pub extend AdvancedDecoderLaunchOperandRole with Eq::{not_equal, equal}

///|
pub extend AdvancedDecoderLaunchOperand with @debug.Debug::{to_repr}

///|
pub extend AdvancedDecoderLaunchOperand with Eq::{not_equal, equal}

///|
pub extend AdvancedDecoderLaunchEvidenceInput with @debug.Debug::{to_repr}

///|
pub extend AdvancedDecoderLaunchEvidenceInput with Eq::{not_equal, equal}

///|
pub extend AdvancedDecoderAdmittedEntryPoint with @debug.Debug::{to_repr}

///|
pub extend AdvancedDecoderAdmittedEntryPoint with Eq::{not_equal, equal}

///|
pub extend AdvancedDecoderAdmittedLaunch with @debug.Debug::{to_repr}

///|
pub extend AdvancedDecoderAdmittedLaunch with Eq::{not_equal, equal}

///|
pub extend AdvancedDecoderArtifactAdmissionDigest with @debug.Debug::{to_repr}

///|
pub extend AdvancedDecoderArtifactAdmissionDigest with Eq::{not_equal, equal}

///|
pub extend AdvancedDecoderArtifactAdmissionDigest with Hash::{
  hash,
  hash_combine,
}

///|
pub extend AdvancedDecoderArtifactAdmissionStatus with @debug.Debug::{to_repr}

///|
pub extend AdvancedDecoderArtifactAdmissionStatus with Eq::{not_equal, equal}
