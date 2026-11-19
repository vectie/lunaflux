///|
pub extend DeviceExecutorStage with @debug.Debug::{to_repr}

///|
pub extend DeviceExecutorStage with Eq::{not_equal, equal}

///|
pub extend DeviceExecutorValidationRule with @debug.Debug::{to_repr}

///|
pub extend DeviceExecutorValidationRule with Eq::{not_equal, equal}

///|
pub extend DeviceExecutorOperationRule with @debug.Debug::{to_repr}

///|
pub extend DeviceExecutorOperationRule with Eq::{not_equal, equal}

///|
pub extend DeviceExecutorCleanupCause with @debug.Debug::{to_repr}

///|
pub extend DeviceExecutorCleanupCause with Eq::{not_equal, equal}

///|
pub extend DeviceExecutorError with @debug.Debug::{to_repr}

///|
pub extend DeviceExecutorError with Eq::{not_equal, equal}
