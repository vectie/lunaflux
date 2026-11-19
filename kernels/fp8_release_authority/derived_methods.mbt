///|
pub extend Fp8ReleaseAuthorityRule with @debug.Debug::{to_repr}

///|
pub extend Fp8ReleaseAuthorityRule with Eq::{not_equal, equal}

///|
pub extend Fp8ReleaseAuthorityError with @debug.Debug::{to_repr}

///|
pub extend Fp8ReleaseAuthorityError with Eq::{not_equal, equal}

///|
pub extend Fp8ReleaseAuthorityDigest with @debug.Debug::{to_repr}

///|
pub extend Fp8ReleaseAuthorityDigest with Eq::{not_equal, equal}

///|
pub extend Fp8ReleaseAuthorityDigest with Hash::{hash, hash_combine}
