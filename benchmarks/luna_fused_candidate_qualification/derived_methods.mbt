///|
pub extend FusedCandidateBenchmarkStage with @debug.Debug::{to_repr}

///|
pub extend FusedCandidateBenchmarkStage with Eq::{not_equal, equal}

///|
pub extend FusedCandidateBenchmarkError with @debug.Debug::{to_repr}

///|
pub extend FusedCandidateBenchmarkError with Eq::{not_equal, equal}

///|
pub extend FusedCandidateBenchmarkPairOrder with @debug.Debug::{to_repr}

///|
pub extend FusedCandidateBenchmarkPairOrder with Eq::{not_equal, equal}

///|
pub extend FusedCandidateBenchmarkShape with @debug.Debug::{to_repr}

///|
pub extend FusedCandidateBenchmarkShape with Eq::{not_equal, equal}
