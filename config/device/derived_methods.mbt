///|
pub extend DeviceField with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend DeviceField with Eq::{not_equal, equal}

///|
pub extend DeviceIssue with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend DeviceIssue with Eq::{not_equal, equal}

///|
pub extend DeviceConfigError with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend DeviceConfigError with Eq::{not_equal, equal}

///|
pub extend DeviceConfig with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend DeviceConfig with Eq::{not_equal, equal}
