///|
pub extend WorkerExecutableDigest with @debug.Debug::{to_repr}

///|
pub extend WorkerExecutableDigest with Eq::{not_equal, equal}

///|
pub extend WorkerExecutableFileError with @debug.Debug::{to_repr}

///|
pub extend WorkerExecutableFileError with Eq::{not_equal, equal}
