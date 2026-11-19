///|
pub extend RootBoundWorkerProcessLifecycle with @debug.Debug::{to_repr}

///|
pub extend RootBoundWorkerProcessLifecycle with Eq::{not_equal, equal}

///|
pub extend RootBoundWorkerProcessFailure with @debug.Debug::{to_repr}

///|
pub extend RootBoundWorkerProcessFailure with Eq::{not_equal, equal}

///|
pub extend RootBoundWorkerProcessError with @debug.Debug::{to_repr}

///|
pub extend RootBoundWorkerProcessError with Eq::{not_equal, equal}

///|
pub extend WorkerProcessLifecycle with @debug.Debug::{to_repr}

///|
pub extend WorkerProcessLifecycle with Eq::{not_equal, equal}

///|
pub extend WorkerProcessRule with @debug.Debug::{to_repr}

///|
pub extend WorkerProcessRule with Eq::{not_equal, equal}

///|
pub extend WorkerTransportFailure with @debug.Debug::{to_repr}

///|
pub extend WorkerTransportFailure with Eq::{not_equal, equal}

///|
pub extend WorkerProcessError with @debug.Debug::{to_repr}

///|
pub extend WorkerProcessError with Eq::{not_equal, equal}
