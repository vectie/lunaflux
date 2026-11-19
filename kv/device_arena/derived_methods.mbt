///|
pub extend DeviceKvArenaRule with @debug.Debug::{to_repr}

///|
pub extend DeviceKvArenaRule with Eq::{not_equal, equal}

///|
pub extend DeviceKvArenaStage with @debug.Debug::{to_repr}

///|
pub extend DeviceKvArenaStage with Eq::{not_equal, equal}

///|
pub extend DeviceKvArenaError with @debug.Debug::{to_repr}

///|
pub extend DeviceKvArenaError with Eq::{not_equal, equal}
