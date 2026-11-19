///|
pub extend IncrementalOutputRule with @debug.Debug::{to_repr}

///|
pub extend IncrementalOutputRule with Eq::{not_equal, equal}

///|
pub extend IncrementalOutputIssue with @debug.Debug::{to_repr}

///|
pub extend IncrementalOutputIssue with Eq::{not_equal, equal}

///|
pub extend IncrementalOutputError with @debug.Debug::{to_repr}

///|
pub extend IncrementalOutputError with Eq::{not_equal, equal}

///|
pub extend LunaIncrementalOutputProgress with @debug.Debug::{to_repr}

///|
pub extend LunaIncrementalOutputProgress with Eq::{not_equal, equal}
