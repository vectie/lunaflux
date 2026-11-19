///|
pub extend RankChildControlRule with @debug.Debug::{to_repr}

///|
pub extend RankChildControlRule with Eq::{not_equal, equal}

///|
pub extend RankChildControlError with @debug.Debug::{to_repr}

///|
pub extend RankChildControlError with Eq::{not_equal, equal}
