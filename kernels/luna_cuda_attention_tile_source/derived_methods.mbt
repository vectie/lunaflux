///|
pub extend CudaPagedAttentionTileAbi with @debug.Debug::{to_repr}

///|
pub extend CudaPagedAttentionTileAbi with Eq::{not_equal, equal}

///|
pub extend CudaAttentionTileSourceDigest with @debug.Debug::{to_repr}

///|
pub extend CudaAttentionTileSourceDigest with Eq::{not_equal, equal}

///|
pub extend CudaAttentionTileSourceDigest with Hash::{hash, hash_combine}

///|
pub extend CudaAttentionTileSource with @debug.Debug::{to_repr}

///|
pub extend CudaAttentionTileSource with Eq::{not_equal, equal}

///|
pub extend CudaPartitionedAttentionWorkspace with @debug.Debug::{to_repr}

///|
pub extend CudaPartitionedAttentionWorkspace with Eq::{not_equal, equal}

///|
pub extend CudaPartitionedAttentionTileSource with @debug.Debug::{to_repr}

///|
pub extend CudaPartitionedAttentionTileSource with Eq::{not_equal, equal}

///|
pub extend CudaAttentionTileSourceRule with @debug.Debug::{to_repr}

///|
pub extend CudaAttentionTileSourceRule with Eq::{not_equal, equal}

///|
pub extend CudaAttentionTileSourceError with @debug.Debug::{to_repr}

///|
pub extend CudaAttentionTileSourceError with Eq::{not_equal, equal}
