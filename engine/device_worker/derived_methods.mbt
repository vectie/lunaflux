///|
pub extend DeviceWorkerRule with @debug.Debug::{to_repr}

///|
pub extend DeviceWorkerRule with Eq::{not_equal, equal}

///|
pub extend DeviceWorkerLifecycle with @debug.Debug::{to_repr}

///|
pub extend DeviceWorkerLifecycle with Eq::{not_equal, equal}

///|
pub extend DeviceWorkerError with @debug.Debug::{to_repr}

///|
pub extend DeviceWorkerError with Eq::{not_equal, equal}
