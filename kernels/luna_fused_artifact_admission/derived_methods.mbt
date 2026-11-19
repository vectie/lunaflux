///|
pub extend DecodeNumericLaw with @debug.Debug::{to_repr}

///|
pub extend DecodeExecutionDescriptor with @debug.Debug::{to_repr}

///|
pub extend PrefillExponentialLaw with @debug.Debug::{to_repr}

///|
pub extend PrefillExponentialLaw with Eq::{not_equal, equal}

///|
pub extend ReusableFusedRuntimeRowVariant with Eq::{not_equal, equal}

///|
pub extend FusedArtifactAdmissionRule with @debug.Debug::{to_repr}

///|
pub extend FusedArtifactAdmissionRule with Eq::{not_equal, equal}

///|
pub extend FusedArtifactAdmissionError with @debug.Debug::{to_repr}

///|
pub extend FusedArtifactAdmissionError with Eq::{not_equal, equal}

///|
pub extend FusedStandaloneFallbackRole with @debug.Debug::{to_repr}

///|
pub extend FusedStandaloneFallbackRole with Eq::{not_equal, equal}

///|
pub extend FusedStandaloneFallbackIdentity with @debug.Debug::{to_repr}

///|
pub extend FusedStandaloneFallbackIdentity with Eq::{not_equal, equal}

///|
pub extend ReusableFusedRuntimeKernelAbi with @debug.Debug::{to_repr}

///|
pub extend ReusableFusedRuntimeKernelAbi with Eq::{not_equal, equal}
