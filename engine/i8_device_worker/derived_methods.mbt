///|
pub extend I8DeviceWorkerRule with @debug.Debug::{to_repr}

///|
pub extend I8DeviceWorkerRule with Eq::{not_equal, equal}

///|
pub extend I8DeviceWorkerLifecycle with @debug.Debug::{to_repr}

///|
pub extend I8DeviceWorkerLifecycle with Eq::{not_equal, equal}

///|
pub extend I8DeviceWorkerError with @debug.Debug::{to_repr}

///|
pub extend I8DeviceWorkerError with Eq::{not_equal, equal}
