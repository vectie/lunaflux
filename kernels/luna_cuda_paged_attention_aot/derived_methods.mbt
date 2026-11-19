///|
pub extend PagedAttentionDecodeSplitCudaBundle with @debug.Debug::{to_repr}

///|
pub extend PagedAttentionDecodeSplitCudaBundle with Eq::{not_equal, equal}

///|
pub extend PagedAttentionDecodeSplitManifestCudaBinding with @debug.Debug::{
  to_repr,
}

///|
pub extend PagedAttentionDecodeSplitManifestCudaBinding with Eq::{
  not_equal,
  equal,
}

///|
pub extend PagedAttentionDecodeSplitRuntimeCudaBinding with @debug.Debug::{
  to_repr,
}

///|
pub extend PagedAttentionDecodeSplitRuntimeCudaBinding with Eq::{
  not_equal,
  equal,
}

///|
pub extend PagedAttentionCudaSourceDigest with @debug.Debug::{to_repr}

///|
pub extend PagedAttentionCudaSourceDigest with Eq::{not_equal, equal}

///|
pub extend PagedAttentionCudaSourceDigest with Hash::{hash, hash_combine}

///|
pub extend PagedAttentionCudaRecipeDigest with @debug.Debug::{to_repr}

///|
pub extend PagedAttentionCudaRecipeDigest with Eq::{not_equal, equal}

///|
pub extend PagedAttentionCudaRecipeDigest with Hash::{hash, hash_combine}

///|
pub extend PagedAttentionManifestBindingDigest with @debug.Debug::{to_repr}

///|
pub extend PagedAttentionManifestBindingDigest with Eq::{not_equal, equal}

///|
pub extend PagedAttentionManifestBindingDigest with Hash::{hash, hash_combine}

///|
pub extend PagedAttentionAotRule with @debug.Debug::{to_repr}

///|
pub extend PagedAttentionAotRule with Eq::{not_equal, equal}

///|
pub extend PagedAttentionAotError with @debug.Debug::{to_repr}

///|
pub extend PagedAttentionAotError with Eq::{not_equal, equal}

///|
pub extend PagedAttentionAotLimits with @debug.Debug::{to_repr}

///|
pub extend PagedAttentionAotLimits with Eq::{not_equal, equal}

///|
pub extend PagedAttentionCudaLowering with @debug.Debug::{to_repr}

///|
pub extend PagedAttentionCudaLowering with Eq::{not_equal, equal}

///|
pub extend PagedAttentionPhaseCudaBundle with @debug.Debug::{to_repr}

///|
pub extend PagedAttentionPhaseCudaBundle with Eq::{not_equal, equal}

///|
pub extend PagedAttentionManifestCudaLowering with @debug.Debug::{to_repr}

///|
pub extend PagedAttentionManifestCudaLowering with Eq::{not_equal, equal}
