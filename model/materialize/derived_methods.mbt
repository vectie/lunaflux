///|
pub extend MaterializationLimit with @debug.Debug::{to_repr}

///|
pub extend MaterializationLimit with Eq::{not_equal, equal}

///|
pub extend MaterializationError with @debug.Debug::{to_repr}

///|
pub extend MaterializationError with Eq::{not_equal, equal}

///|
pub extend MaterializationLimits with @debug.Debug::{to_repr}

///|
pub extend MaterializationLimits with Eq::{not_equal, equal}

///|
pub extend TensorChunk with @debug.Debug::{to_repr}

///|
pub extend TensorChunk with Eq::{not_equal, equal}

///|
pub extend TensorReceipt with @debug.Debug::{to_repr}

///|
pub extend TensorReceipt with Eq::{not_equal, equal}

///|
pub extend MaterializationReport with @debug.Debug::{to_repr}

///|
pub extend MaterializationReport with Eq::{not_equal, equal}

///|
pub extend HostTensor with @debug.Debug::{to_repr}

///|
pub extend HostTensor with Eq::{not_equal, equal}

///|
pub extend HostMaterialization with @debug.Debug::{to_repr}

///|
pub extend HostMaterialization with Eq::{not_equal, equal}
