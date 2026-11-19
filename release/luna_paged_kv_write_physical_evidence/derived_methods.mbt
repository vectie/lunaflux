///|
pub extend PagedKvWritePhysicalRule with @debug.Debug::{to_repr}

///|
pub extend PagedKvWritePhysicalRule with Eq::{not_equal, equal}

///|
pub extend PagedKvWritePhysicalError with @debug.Debug::{to_repr}

///|
pub extend PagedKvWritePhysicalError with Eq::{not_equal, equal}
