///|
pub extend KernelArtifactLimit with @debug.Debug::{to_repr}

///|
pub extend KernelArtifactLimit with Eq::{not_equal, equal}

///|
pub extend FunctionSymbolRule with @debug.Debug::{to_repr}

///|
pub extend FunctionSymbolRule with Eq::{not_equal, equal}

///|
pub extend KernelArtifactError with @debug.Debug::{to_repr}

///|
pub extend KernelArtifactError with Eq::{not_equal, equal}

///|
pub extend KernelRowVariant with @debug.Debug::{to_repr}

///|
pub extend KernelRowVariant with Eq::{not_equal, equal}
