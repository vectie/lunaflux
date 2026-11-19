///|
pub extend ModelConfigFileLimit with @debug.Debug::{to_repr}

///|
pub extend ModelConfigFileLimit with Eq::{not_equal, equal}

///|
pub extend ModelConfigFileOperation with @debug.Debug::{to_repr}

///|
pub extend ModelConfigFileOperation with Eq::{not_equal, equal}

///|
pub extend ModelConfigFileIoIssue with @debug.Debug::{to_repr}

///|
pub extend ModelConfigFileIoIssue with Eq::{not_equal, equal}

///|
pub extend ModelConfigFileError with @debug.Debug::{to_repr}

///|
pub extend ModelConfigFileError with Eq::{not_equal, equal}

///|
pub extend ModelConfigDigest with @debug.Debug::{to_repr}

///|
pub extend ModelConfigDigest with Eq::{not_equal, equal}
