///|
pub extend RuntimeInstanceAdmissionStage with @debug.Debug::{to_repr}

///|
pub extend RuntimeInstanceAdmissionStage with Eq::{not_equal, equal}

///|
pub extend RuntimeInstanceAdmissionError with @debug.Debug::{to_repr}

///|
pub extend RuntimeInstanceAdmissionError with Eq::{not_equal, equal}
