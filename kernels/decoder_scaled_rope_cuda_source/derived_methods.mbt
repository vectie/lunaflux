///|
pub extend DecoderScaledRopeNumericalContract with @debug.Debug::{to_repr}

///|
pub extend DecoderScaledRopeNumericalContract with Eq::{not_equal, equal}

///|
pub extend DecoderScaledRopeCudaSourceError with @debug.Debug::{to_repr}

///|
pub extend DecoderScaledRopeCudaSourceError with Eq::{not_equal, equal}
