///|
pub extend ProcessLimitRule with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ProcessLimitRule with Eq::{not_equal, equal}

///|
pub extend ProcessOperation with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ProcessOperation with Eq::{not_equal, equal}

///|
pub extend ProcessError with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ProcessError with Eq::{not_equal, equal}
