///|
pub extend DecoderHyperConnectionCudaOperation with @debug.Debug::{to_repr}

///|
pub extend DecoderHyperConnectionCudaOperation with Eq::{not_equal, equal}

///|
pub extend DecoderHyperConnectionNumericalContract with @debug.Debug::{to_repr}

///|
pub extend DecoderHyperConnectionNumericalContract with Eq::{not_equal, equal}

///|
pub extend DecoderHyperConnectionCudaSourceError with @debug.Debug::{to_repr}

///|
pub extend DecoderHyperConnectionCudaSourceError with Eq::{not_equal, equal}
