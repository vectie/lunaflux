///|
pub extend Fp8KernelCapabilityManifestVersion with @debug.Debug::{to_repr}

///|
pub extend Fp8KernelCapabilityManifestVersion with Eq::{not_equal, equal}

///|
pub extend Fp8KernelExecutionScope with @debug.Debug::{to_repr}

///|
pub extend Fp8KernelExecutionScope with Eq::{not_equal, equal}

///|
pub extend Fp8KernelCapabilityManifestDigest with @debug.Debug::{to_repr}

///|
pub extend Fp8KernelCapabilityManifestDigest with Eq::{not_equal, equal}

///|
pub extend Fp8KernelCapabilityManifestDigest with Hash::{hash, hash_combine}

///|
pub extend Fp8KernelManifestClaim with @debug.Debug::{to_repr}

///|
pub extend Fp8KernelManifestClaim with Eq::{not_equal, equal}

///|
pub extend Fp8KernelWeightBinding with @debug.Debug::{to_repr}

///|
pub extend Fp8KernelWeightBinding with Eq::{not_equal, equal}

///|
pub extend Fp8KernelManifestEntry with @debug.Debug::{to_repr}

///|
pub extend Fp8KernelManifestEntry with Eq::{not_equal, equal}

///|
pub extend Fp8KernelCapabilityManifest with @debug.Debug::{to_repr}

///|
pub extend Fp8KernelCapabilityManifest with Eq::{not_equal, equal}

///|
pub extend Fp8KernelManifestLimits with @debug.Debug::{to_repr}

///|
pub extend Fp8KernelManifestLimits with Eq::{not_equal, equal}

///|
pub extend Fp8KernelManifestRule with @debug.Debug::{to_repr}

///|
pub extend Fp8KernelManifestRule with Eq::{not_equal, equal}

///|
pub extend Fp8KernelManifestError with @debug.Debug::{to_repr}

///|
pub extend Fp8KernelManifestError with Eq::{not_equal, equal}
