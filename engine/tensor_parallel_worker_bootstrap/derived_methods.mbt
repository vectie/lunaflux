///|
pub extend TensorParallelWorkerBootstrapRule with @debug.Debug::{to_repr}

///|
pub extend TensorParallelWorkerBootstrapRule with Eq::{not_equal, equal}

///|
pub extend TensorParallelWorkerBootstrapError with @debug.Debug::{to_repr}

///|
pub extend TensorParallelWorkerBootstrapError with Eq::{not_equal, equal}

///|
pub extend TensorParallelGroupBootstrapDigest with @debug.Debug::{to_repr}

///|
pub extend TensorParallelGroupBootstrapDigest with Eq::{not_equal, equal}

///|
pub extend TensorParallelTopologyDigest with @debug.Debug::{to_repr}

///|
pub extend TensorParallelTopologyDigest with Eq::{not_equal, equal}

///|
pub extend TensorParallelWorkerContractDigest with @debug.Debug::{to_repr}

///|
pub extend TensorParallelWorkerContractDigest with Eq::{not_equal, equal}

///|
pub extend TensorParallelDevicePlanDigest with @debug.Debug::{to_repr}

///|
pub extend TensorParallelDevicePlanDigest with Eq::{not_equal, equal}

///|
pub extend TensorParallelKvBindingDigest with @debug.Debug::{to_repr}

///|
pub extend TensorParallelKvBindingDigest with Eq::{not_equal, equal}
