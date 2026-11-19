///|
pub extend DeepSeekV4TokenHashMaterializationError with @debug.Debug::{to_repr}

///|
pub extend DeepSeekV4TokenHashMaterializationError with Eq::{not_equal, equal}

///|
pub extend DeepSeekV4HostExecutionLimitation with @debug.Debug::{to_repr}

///|
pub extend DeepSeekV4HostExecutionLimitation with Eq::{not_equal, equal}

///|
pub extend DeepSeekV4HostMaterializationReadiness with @debug.Debug::{to_repr}

///|
pub extend DeepSeekV4HostMaterializationReadiness with Eq::{not_equal, equal}

///|
pub extend DeepSeekV4HostMaterializationError with @debug.Debug::{to_repr}

///|
pub extend DeepSeekV4HostMaterializationError with Eq::{not_equal, equal}

///|
pub extend DeepSeekV4HostExpectedDigests with @debug.Debug::{to_repr}

///|
pub extend DeepSeekV4HostExpectedDigests with Eq::{not_equal, equal}

///|
pub extend DeepSeekV4TensorIndexInput with @debug.Debug::{to_repr}

///|
pub extend DeepSeekV4TensorIndexInput with Eq::{not_equal, equal}

///|
pub extend DeepSeekV4HostManifestDigest with @debug.Debug::{to_repr}

///|
pub extend DeepSeekV4HostManifestDigest with Eq::{not_equal, equal}

///|
pub extend DeepSeekV4HostManifestDigest with Hash::{hash, hash_combine}

///|
pub extend DeepSeekV4HostTensor with @debug.Debug::{to_repr}

///|
pub extend DeepSeekV4HostTensor with Eq::{not_equal, equal}
