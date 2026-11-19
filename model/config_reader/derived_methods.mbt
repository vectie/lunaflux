///|
pub extend ModelConfigField with @debug.Debug::{to_repr}

///|
pub extend ModelConfigField with Eq::{not_equal, equal}

///|
pub extend ExecutableField with @debug.Debug::{to_repr}

///|
pub extend ExecutableField with Eq::{not_equal, equal}

///|
pub extend ModelConfigValueIssue with @debug.Debug::{to_repr}

///|
pub extend ModelConfigValueIssue with Eq::{not_equal, equal}

///|
pub extend ModelConfigReaderError with @debug.Debug::{to_repr}

///|
pub extend ModelConfigReaderError with Eq::{not_equal, equal}
