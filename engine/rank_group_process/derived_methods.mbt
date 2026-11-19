///|
pub extend RankGroupProcessRule with @debug.Debug::{to_repr}

///|
pub extend RankGroupProcessRule with Eq::{not_equal, equal}

///|
pub extend RankGroupProcessFailure with @debug.Debug::{to_repr}

///|
pub extend RankGroupProcessFailure with Eq::{not_equal, equal}

///|
pub extend RankGroupProcessCleanupFailure with @debug.Debug::{to_repr}

///|
pub extend RankGroupProcessCleanupFailure with Eq::{not_equal, equal}

///|
pub extend RankGroupProcessError with @debug.Debug::{to_repr}

///|
pub extend RankGroupProcessError with Eq::{not_equal, equal}

///|
pub extend RankGroupProcessLifecycle with @debug.Debug::{to_repr}

///|
pub extend RankGroupProcessLifecycle with Eq::{not_equal, equal}

///|
pub extend RankGroupProcessProgress with @debug.Debug::{to_repr}

///|
pub extend RankGroupProcessProgress with Eq::{not_equal, equal}

///|
pub extend RankGroupProcessCleanupProgress with @debug.Debug::{to_repr}

///|
pub extend RankGroupProcessCleanupProgress with Eq::{not_equal, equal}
