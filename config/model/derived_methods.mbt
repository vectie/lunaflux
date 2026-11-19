///|
pub extend ModelField with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ModelField with Eq::{not_equal, equal}

///|
pub extend ModelIssue with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ModelIssue with Eq::{not_equal, equal}

///|
pub extend ModelConfigError with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ModelConfigError with Eq::{not_equal, equal}

///|
pub extend ModelConfig with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ModelConfig with Eq::{not_equal, equal}
