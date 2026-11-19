///|
pub extend DeviceWorkerChildError with @debug.Debug::{to_repr}

///|
pub extend DeviceWorkerChildError with Eq::{not_equal, equal}
