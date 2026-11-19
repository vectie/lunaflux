///|
pub extend ExecutionLimitField with @debug.Debug::{to_repr}

///|
pub extend ExecutionLimitField with Eq::{not_equal, equal}

///|
pub extend ExecutionGraphFailure with @debug.Debug::{to_repr}

///|
pub extend ExecutionGraphFailure with Eq::{not_equal, equal}

///|
pub extend ReferenceExecutionError with @debug.Debug::{to_repr}

///|
pub extend ReferenceExecutionError with Eq::{not_equal, equal}

///|
pub extend ExecutionLimits with @debug.Debug::{to_repr}

///|
pub extend ExecutionLimits with Eq::{not_equal, equal}

///|
pub extend ExecutionResult with @debug.Debug::{to_repr}

///|
pub extend ExecutionResult with Eq::{not_equal, equal}

///|
pub extend PreparedModel with @debug.Debug::{to_repr}

///|
pub extend PreparedModel with Eq::{not_equal, equal}

///|
pub extend GenerationResult with @debug.Debug::{to_repr}

///|
pub extend GenerationResult with Eq::{not_equal, equal}
