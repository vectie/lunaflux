///|
pub extend Fp8StartupAdmissionVersion with @debug.Debug::{to_repr}

///|
pub extend Fp8StartupAdmissionVersion with Eq::{not_equal, equal}

///|
pub extend Fp8StartupAdmissionDigest with @debug.Debug::{to_repr}

///|
pub extend Fp8StartupAdmissionDigest with Eq::{not_equal, equal}

///|
pub extend Fp8StartupAdmissionDigest with Hash::{hash, hash_combine}

///|
pub extend Fp8StartupAdmissionLimits with @debug.Debug::{to_repr}

///|
pub extend Fp8StartupAdmissionLimits with Eq::{not_equal, equal}

///|
pub extend Fp8StartupAdmissionRule with @debug.Debug::{to_repr}

///|
pub extend Fp8StartupAdmissionRule with Eq::{not_equal, equal}

///|
pub extend Fp8StartupAdmissionError with @debug.Debug::{to_repr}

///|
pub extend Fp8StartupAdmissionError with Eq::{not_equal, equal}

///|
pub extend Fp8StartupAdmissionV2Version with @debug.Debug::{to_repr}

///|
pub extend Fp8StartupAdmissionV2Version with Eq::{not_equal, equal}

///|
pub extend Fp8StartupAdmissionV2Digest with @debug.Debug::{to_repr}

///|
pub extend Fp8StartupAdmissionV2Digest with Eq::{not_equal, equal}

///|
pub extend Fp8StartupAdmissionV2Digest with Hash::{hash, hash_combine}
