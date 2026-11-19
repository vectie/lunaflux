///|
pub extend PagedKvWriteDigest with @debug.Debug::{to_repr}

///|
pub extend PagedKvWriteDigest with Eq::{not_equal, equal}

///|
pub extend PagedKvWriteDigest with Hash::{hash, hash_combine}

///|
pub extend PagedKvWriteAotRule with @debug.Debug::{to_repr}

///|
pub extend PagedKvWriteAotRule with Eq::{not_equal, equal}

///|
pub extend PagedKvWriteAotError with @debug.Debug::{to_repr}

///|
pub extend PagedKvWriteAotError with Eq::{not_equal, equal}

///|
pub extend PagedKvWriteKernelAbi with @debug.Debug::{to_repr}

///|
pub extend PagedKvWriteKernelAbi with Eq::{not_equal, equal}
