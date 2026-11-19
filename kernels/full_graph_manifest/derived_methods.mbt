///|
pub extend FullGraphManifestRule with @debug.Debug::{to_repr}

///|
pub extend FullGraphManifestRule with Eq::{not_equal, equal}

///|
pub extend FullGraphManifestError with @debug.Debug::{to_repr}

///|
pub extend FullGraphManifestError with Eq::{not_equal, equal}
