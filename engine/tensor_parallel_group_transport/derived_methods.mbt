///|
pub extend TensorParallelGroupTransportRule with @debug.Debug::{to_repr}

///|
pub extend TensorParallelGroupTransportRule with Eq::{not_equal, equal}

///|
pub extend TensorParallelGroupTransportError with @debug.Debug::{to_repr}

///|
pub extend TensorParallelGroupTransportError with Eq::{not_equal, equal}

///|
pub extend TensorParallelGroupTransportLifecycle with @debug.Debug::{to_repr}

///|
pub extend TensorParallelGroupTransportLifecycle with Eq::{not_equal, equal}

///|
pub extend TensorParallelGroupStart with @debug.Debug::{to_repr}

///|
pub extend TensorParallelGroupStart with Eq::{not_equal, equal}

///|
pub extend TensorParallelGroupCleanupProgress with @debug.Debug::{to_repr}

///|
pub extend TensorParallelGroupCleanupProgress with Eq::{not_equal, equal}

///|
pub extend TensorParallelGroupTemplateDigest with @debug.Debug::{to_repr}

///|
pub extend TensorParallelGroupTemplateDigest with Eq::{not_equal, equal}
