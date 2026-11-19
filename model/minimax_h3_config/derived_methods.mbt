///|
pub extend MiniMaxH3ConfigDocument with @debug.Debug::{to_repr}

///|
pub extend MiniMaxH3ConfigDocument with Eq::{not_equal, equal}

///|
pub extend MiniMaxH3ConfigError with @debug.Debug::{to_repr}

///|
pub extend MiniMaxH3ConfigError with Eq::{not_equal, equal}

///|
pub extend MiniMaxH3PipelineIndex with @debug.Debug::{to_repr}

///|
pub extend MiniMaxH3PipelineIndex with Eq::{not_equal, equal}

///|
pub extend MiniMaxH3VideoVaeConfig with @debug.Debug::{to_repr}

///|
pub extend MiniMaxH3VideoVaeConfig with Eq::{not_equal, equal}

///|
pub extend MiniMaxH3AudioVaeConfig with @debug.Debug::{to_repr}

///|
pub extend MiniMaxH3AudioVaeConfig with Eq::{not_equal, equal}

///|
pub extend MiniMaxH3SchedulerConfig with @debug.Debug::{to_repr}

///|
pub extend MiniMaxH3SchedulerConfig with Eq::{not_equal, equal}
