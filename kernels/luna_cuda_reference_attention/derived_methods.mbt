///|
pub extend ReferenceAttentionBackend with @debug.Debug::{to_repr}

///|
pub extend ReferenceAttentionBackend with Eq::{not_equal, equal}

///|
pub extend ReferenceAttentionPlan with @debug.Debug::{to_repr}

///|
pub extend ReferenceAttentionPlan with Eq::{not_equal, equal}

///|
pub extend ReferenceAttentionError with @debug.Debug::{to_repr}

///|
pub extend ReferenceAttentionError with Eq::{not_equal, equal}

///|
pub extend ReferencePrefillSchedule with @debug.Debug::{to_repr}

///|
pub extend ReferencePrefillSchedule with Eq::{not_equal, equal}
