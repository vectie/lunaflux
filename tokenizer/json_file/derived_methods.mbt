///|
pub extend TokenizerFileLimit with @debug.Debug::{to_repr}

///|
pub extend TokenizerFileLimit with Eq::{not_equal, equal}

///|
pub extend TokenizerFileOperation with @debug.Debug::{to_repr}

///|
pub extend TokenizerFileOperation with Eq::{not_equal, equal}

///|
pub extend TokenizerFileError with @debug.Debug::{to_repr}

///|
pub extend TokenizerFileError with Eq::{not_equal, equal}
