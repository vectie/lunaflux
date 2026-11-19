///|
pub extend CudaAttentionComputePrimitive with @debug.Debug::{to_repr}

///|
pub extend CudaAttentionComputePrimitive with Eq::{not_equal, equal}

///|
pub extend CudaAttentionCopyPrimitive with @debug.Debug::{to_repr}

///|
pub extend CudaAttentionCopyPrimitive with Eq::{not_equal, equal}

///|
pub extend CudaAttentionTileLoweringDigest with @debug.Debug::{to_repr}

///|
pub extend CudaAttentionTileLoweringDigest with Eq::{not_equal, equal}

///|
pub extend CudaAttentionTileLoweringDigest with Hash::{hash, hash_combine}

///|
pub extend CudaAttentionTileLowering with @debug.Debug::{to_repr}

///|
pub extend CudaAttentionTileLowering with Eq::{not_equal, equal}

///|
pub extend CudaAttentionTileLoweringRule with @debug.Debug::{to_repr}

///|
pub extend CudaAttentionTileLoweringRule with Eq::{not_equal, equal}

///|
pub extend CudaAttentionTileLoweringError with @debug.Debug::{to_repr}

///|
pub extend CudaAttentionTileLoweringError with Eq::{not_equal, equal}
