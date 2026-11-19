///|
pub extend TensorParallelDeviceWorkerRule with @debug.Debug::{to_repr}

///|
pub extend TensorParallelDeviceWorkerRule with Eq::{not_equal, equal}

///|
pub extend TensorParallelDeviceWorkerResource with @debug.Debug::{to_repr}

///|
pub extend TensorParallelDeviceWorkerResource with Eq::{not_equal, equal}

///|
pub extend TensorParallelDeviceWorkerFailureClass with @debug.Debug::{to_repr}

///|
pub extend TensorParallelDeviceWorkerFailureClass with Eq::{not_equal, equal}

///|
pub extend TensorParallelDeviceWorkerError with @debug.Debug::{to_repr}

///|
pub extend TensorParallelDeviceWorkerError with Eq::{not_equal, equal}

///|
pub extend TensorParallelDeviceWorkerLifecycle with @debug.Debug::{to_repr}

///|
pub extend TensorParallelDeviceWorkerLifecycle with Eq::{not_equal, equal}
