///|
pub extend LunaKernelRootRule with @debug.Debug::{to_repr}

///|
pub extend LunaKernelRootRule with Eq::{not_equal, equal}

///|
pub extend LunaKernelRootError with @debug.Debug::{to_repr}

///|
pub extend LunaKernelRootError with Eq::{not_equal, equal}
