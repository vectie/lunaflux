///|
pub extend LunaKernelBundleRule with @debug.Debug::{to_repr}

///|
pub extend LunaKernelBundleRule with Eq::{not_equal, equal}

///|
pub extend LunaKernelBundleError with @debug.Debug::{to_repr}

///|
pub extend LunaKernelBundleError with Eq::{not_equal, equal}

///|
pub extend LunaDeterministicCompileReceipt with @debug.Debug::{to_repr}

///|
pub extend LunaDeterministicCompileReceipt with Eq::{not_equal, equal}
