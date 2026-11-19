///|
pub extend SchemaVersion with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend SchemaVersion with Eq::{not_equal, equal}

///|
pub extend Section with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend Section with Eq::{not_equal, equal}

///|
pub extend SchemaError with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend SchemaError with Eq::{not_equal, equal}
