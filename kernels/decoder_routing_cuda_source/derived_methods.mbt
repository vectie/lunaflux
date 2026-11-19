///|
pub extend DecoderRoutingCudaOperation with @debug.Debug::{to_repr}

///|
pub extend DecoderRoutingCudaOperation with Eq::{not_equal, equal}

///|
pub extend DecoderRoutingNumericalContract with @debug.Debug::{to_repr}

///|
pub extend DecoderRoutingNumericalContract with Eq::{not_equal, equal}

///|
pub extend DecoderRoutingCudaSourceError with @debug.Debug::{to_repr}

///|
pub extend DecoderRoutingCudaSourceError with Eq::{not_equal, equal}
