///|
pub extend QkNormRopePagedKvWriteDigest with @debug.Debug::{to_repr}

///|
pub extend QkNormRopePagedKvWriteDigest with Eq::{not_equal, equal}

///|
pub extend QkNormRopePagedKvWriteDigest with Hash::{hash, hash_combine}

///|
pub extend QkNormRopePagedKvWriteAotRule with @debug.Debug::{to_repr}

///|
pub extend QkNormRopePagedKvWriteAotRule with Eq::{not_equal, equal}

///|
pub extend QkNormRopePagedKvWriteAotError with @debug.Debug::{to_repr}

///|
pub extend QkNormRopePagedKvWriteAotError with Eq::{not_equal, equal}

///|
pub extend QkNormRopePagedKvWriteKernelAbi with @debug.Debug::{to_repr}

///|
pub extend QkNormRopePagedKvWriteKernelAbi with Eq::{not_equal, equal}
