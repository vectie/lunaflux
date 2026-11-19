///|
pub extend FusedParallelKernelAbi with @debug.Debug::{to_repr}

///|
pub extend FusedParallelKernelAbi with Eq::{not_equal, equal}

///|
pub extend FusedParallelArtifactPurpose with @debug.Debug::{to_repr}

///|
pub extend FusedParallelArtifactPurpose with Eq::{not_equal, equal}

///|
pub extend FusedParallelDigest with @debug.Debug::{to_repr}

///|
pub extend FusedParallelDigest with Eq::{not_equal, equal}

///|
pub extend FusedParallelDigest with Hash::{hash, hash_combine}

///|
pub extend FusedParallelAotRule with @debug.Debug::{to_repr}

///|
pub extend FusedParallelAotRule with Eq::{not_equal, equal}

///|
pub extend FusedParallelAotError with @debug.Debug::{to_repr}

///|
pub extend FusedParallelAotError with Eq::{not_equal, equal}

///|
pub extend FusedParallelFamily with @debug.Debug::{to_repr}

///|
pub extend FusedParallelFamily with Eq::{not_equal, equal}

///|
pub extend FusedParallelNumericalPolicy with @debug.Debug::{to_repr}

///|
pub extend FusedParallelNumericalPolicy with Eq::{not_equal, equal}
