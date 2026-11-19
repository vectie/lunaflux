///|
pub extend PagedAttentionParallelAotRule with @debug.Debug::{to_repr}

///|
pub extend PagedAttentionParallelAotRule with Eq::{not_equal, equal}

///|
pub extend PagedAttentionParallelAotError with @debug.Debug::{to_repr}

///|
pub extend PagedAttentionParallelAotError with Eq::{not_equal, equal}

///|
pub extend PagedAttentionParallelDigest with @debug.Debug::{to_repr}

///|
pub extend PagedAttentionParallelDigest with Eq::{not_equal, equal}

///|
pub extend PagedAttentionParallelDigest with Hash::{hash, hash_combine}
