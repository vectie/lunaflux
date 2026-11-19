///|
pub extend PagedAttentionReadOnlyDigest with @debug.Debug::{to_repr}

///|
pub extend PagedAttentionReadOnlyDigest with Eq::{not_equal, equal}

///|
pub extend PagedAttentionReadOnlyDigest with Hash::{hash, hash_combine}

///|
pub extend PagedAttentionReadOnlyRule with @debug.Debug::{to_repr}

///|
pub extend PagedAttentionReadOnlyRule with Eq::{not_equal, equal}

///|
pub extend PagedAttentionReadOnlyKernelAbi with @debug.Debug::{to_repr}

///|
pub extend PagedAttentionReadOnlyKernelAbi with Eq::{not_equal, equal}
