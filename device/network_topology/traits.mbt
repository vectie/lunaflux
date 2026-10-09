///|
pub extend NetworkTopologyRule with Eq::{equal, not_equal}

///|
pub extend NetworkTopologyRule with @debug.Debug::{to_repr}

///|
pub extend NetworkTopologyError with Eq::{equal, not_equal}

///|
pub extend NetworkTopologyError with @debug.Debug::{to_repr}

///|
pub extend NodeIdentity with Eq::{equal, not_equal}

///|
pub extend NodeIdentity with @debug.Debug::{to_repr}

///|
pub extend NetworkRank with Eq::{equal, not_equal}

///|
pub extend NetworkRank with @debug.Debug::{to_repr}

///|
pub extend TwoNodeTopology with Eq::{equal, not_equal}

///|
pub extend TwoNodeTopology with @debug.Debug::{to_repr}
