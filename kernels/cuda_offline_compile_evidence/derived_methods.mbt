///|
pub extend CudaOfflineCompileDigestKind with @debug.Debug::{to_repr}

///|
pub extend CudaOfflineCompileDigestKind with Eq::{not_equal, equal}

///|
pub extend CudaOfflineCompileEvidenceStatus with @debug.Debug::{to_repr}

///|
pub extend CudaOfflineCompileEvidenceStatus with Eq::{not_equal, equal}

///|
pub extend CudaOfflineCompileEvidenceDigest with @debug.Debug::{to_repr}

///|
pub extend CudaOfflineCompileEvidenceDigest with Eq::{not_equal, equal}

///|
pub extend CudaOfflineCompileEvidenceDigest with Hash::{hash, hash_combine}

///|
pub extend CudaOfflineCompileEvidenceError with @debug.Debug::{to_repr}

///|
pub extend CudaOfflineCompileEvidenceError with Eq::{not_equal, equal}
