///|
pub extend ReadOnlyProductionDigest with @debug.Debug::{to_repr}

///|
pub extend ReadOnlyProductionDigest with Eq::{not_equal, equal}

///|
pub extend ReadOnlyProductionDigest with Hash::{hash, hash_combine}

///|
pub extend ReadOnlyProductionRule with @debug.Debug::{to_repr}

///|
pub extend ReadOnlyProductionRule with Eq::{not_equal, equal}

///|
pub extend ReadOnlyProductionError with @debug.Debug::{to_repr}

///|
pub extend ReadOnlyProductionError with Eq::{not_equal, equal}
