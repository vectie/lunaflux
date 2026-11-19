///|
pub extend SamplingBoundaryRule with @debug.Debug::{to_repr}

///|
pub extend SamplingBoundaryRule with Eq::{not_equal, equal}

///|
pub extend SamplingBoundaryError with @debug.Debug::{to_repr}

///|
pub extend SamplingBoundaryError with Eq::{not_equal, equal}

///|
pub extend GreedySamplingAotLimits with @debug.Debug::{to_repr}

///|
pub extend GreedySamplingAotLimits with Eq::{not_equal, equal}

///|
pub extend GreedySamplingCudaLowering with @debug.Debug::{to_repr}

///|
pub extend GreedySamplingCudaLowering with Eq::{not_equal, equal}

///|
pub extend GreedySamplingCompileReceipt with @debug.Debug::{to_repr}

///|
pub extend GreedySamplingCompileReceipt with Eq::{not_equal, equal}

///|
pub extend GreedySamplingCudaArtifact with @debug.Debug::{to_repr}

///|
pub extend GreedySamplingCudaArtifact with Eq::{not_equal, equal}

///|
pub extend DeviceSamplingAotLimits with @debug.Debug::{to_repr}

///|
pub extend DeviceSamplingAotLimits with Eq::{not_equal, equal}

///|
pub extend DeviceSamplingCudaLowering with @debug.Debug::{to_repr}

///|
pub extend DeviceSamplingCudaLowering with Eq::{not_equal, equal}

///|
pub extend DeviceSamplingCompileReceipt with @debug.Debug::{to_repr}

///|
pub extend DeviceSamplingCompileReceipt with Eq::{not_equal, equal}

///|
pub extend DeviceSamplingCudaArtifact with @debug.Debug::{to_repr}

///|
pub extend DeviceSamplingCudaArtifact with Eq::{not_equal, equal}

///|
pub extend DeviceSamplingCudaOperandV2 with @debug.Debug::{to_repr}

///|
pub extend DeviceSamplingCudaOperandV2 with Eq::{not_equal, equal}

///|
pub extend DeviceSamplingCudaAbiV2 with @debug.Debug::{to_repr}

///|
pub extend DeviceSamplingCudaAbiV2 with Eq::{not_equal, equal}

///|
pub extend SamplingExecutionOwner with @debug.Debug::{to_repr}

///|
pub extend SamplingExecutionOwner with Eq::{not_equal, equal}

///|
pub extend HostSamplingBoundary with @debug.Debug::{to_repr}

///|
pub extend HostSamplingBoundary with Eq::{not_equal, equal}
