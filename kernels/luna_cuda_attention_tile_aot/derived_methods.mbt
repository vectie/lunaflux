///|
pub extend CudaAttentionTileAotRequest with @debug.Debug::{to_repr}

///|
pub extend CudaAttentionTileAotRequest with Eq::{not_equal, equal}

///|
pub extend CudaAttentionTileAotDigest with @debug.Debug::{to_repr}

///|
pub extend CudaAttentionTileAotDigest with Eq::{not_equal, equal}

///|
pub extend CudaAttentionTileAotDigest with Hash::{hash, hash_combine}

///|
pub extend CudaAttentionTileAotCompilation with @debug.Debug::{to_repr}

///|
pub extend CudaAttentionTileAotCompilation with Eq::{not_equal, equal}

///|
pub extend CudaAttentionTileAotFrontier with @debug.Debug::{to_repr}

///|
pub extend CudaAttentionTileAotFrontier with Eq::{not_equal, equal}

///|
pub extend CudaPartitionedAttentionTileAotRequest with @debug.Debug::{to_repr}

///|
pub extend CudaPartitionedAttentionTileAotRequest with Eq::{not_equal, equal}

///|
pub extend CudaPartitionedAttentionTileAotCompilation with @debug.Debug::{
  to_repr,
}

///|
pub extend CudaPartitionedAttentionTileAotCompilation with Eq::{
  not_equal,
  equal,
}

///|
pub extend CudaPartitionedAttentionTileAotVariant with @debug.Debug::{to_repr}

///|
pub extend CudaPartitionedAttentionTileAotVariant with Eq::{not_equal, equal}

///|
pub extend CudaPartitionedAttentionTileAotFrontier with @debug.Debug::{to_repr}

///|
pub extend CudaPartitionedAttentionTileAotFrontier with Eq::{not_equal, equal}

///|
pub extend CudaAttentionTileAotRule with @debug.Debug::{to_repr}

///|
pub extend CudaAttentionTileAotRule with Eq::{not_equal, equal}

///|
pub extend CudaAttentionTileAotError with @debug.Debug::{to_repr}

///|
pub extend CudaAttentionTileAotError with Eq::{not_equal, equal}
