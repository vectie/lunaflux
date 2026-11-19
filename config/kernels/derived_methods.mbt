///|
pub extend KernelConfigField with @debug.Debug::{to_repr}

///|
pub extend KernelConfigField with Eq::{not_equal, equal}

///|
pub extend KernelConfigIssue with @debug.Debug::{to_repr}

///|
pub extend KernelConfigIssue with Eq::{not_equal, equal}

///|
pub extend KernelConfigError with @debug.Debug::{to_repr}

///|
pub extend KernelConfigError with Eq::{not_equal, equal}

///|
pub extend KernelAdmissionPolicy with @debug.Debug::{to_repr}

///|
pub extend KernelAdmissionPolicy with Eq::{not_equal, equal}

///|
pub extend KernelManifestDigest with @debug.Debug::{to_repr}

///|
pub extend KernelManifestDigest with Eq::{not_equal, equal}

///|
pub extend KernelManifestDigest with Hash::{hash, hash_combine}
