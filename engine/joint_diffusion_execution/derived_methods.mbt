///|
pub extend PipelineStage with Eq::{not_equal, equal}

///|
pub extend PipelineStage with @debug.Debug::{to_repr}

///|
pub extend PreparedStage with PreparedFrame::{execute, drain, release}
