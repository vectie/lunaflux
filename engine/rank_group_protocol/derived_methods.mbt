///|
pub extend RankGroupLifecycle with @debug.Debug::{to_repr}

///|
pub extend RankGroupLifecycle with Eq::{not_equal, equal}

///|
pub extend RankGroupRule with @debug.Debug::{to_repr}

///|
pub extend RankGroupRule with Eq::{not_equal, equal}

///|
pub extend RankGroupFailureReason with @debug.Debug::{to_repr}

///|
pub extend RankGroupFailureReason with Eq::{not_equal, equal}

///|
pub extend RankGroupError with @debug.Debug::{to_repr}

///|
pub extend RankGroupError with Eq::{not_equal, equal}

///|
pub extend RankGroupGeneration with @debug.Debug::{to_repr}

///|
pub extend RankGroupGeneration with Eq::{not_equal, equal}

///|
pub extend RankGroupGeneration with Hash::{hash, hash_combine}
