///|
pub extend PagedCompositeArtifactRule with @debug.Debug::{to_repr}

///|
pub extend PagedCompositeArtifactRule with Eq::{not_equal, equal}

///|
pub extend PagedCompositeArtifactError with @debug.Debug::{to_repr}

///|
pub extend PagedCompositeArtifactError with Eq::{not_equal, equal}
