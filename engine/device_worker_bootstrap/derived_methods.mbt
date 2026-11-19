///|
pub extend DeviceWorkerBootstrapRule with @debug.Debug::{to_repr}

///|
pub extend DeviceWorkerBootstrapRule with Eq::{not_equal, equal}

///|
pub extend BootstrapRootRole with @debug.Debug::{to_repr}

///|
pub extend BootstrapRootRole with Eq::{not_equal, equal}

///|
pub extend BootstrapRootIssue with @debug.Debug::{to_repr}

///|
pub extend BootstrapRootIssue with Eq::{not_equal, equal}

///|
pub extend DeviceWorkerBootstrapError with @debug.Debug::{to_repr}

///|
pub extend DeviceWorkerBootstrapError with Eq::{not_equal, equal}
