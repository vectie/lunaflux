///|
pub extend DecoderFoundationCudaOperation with @debug.Debug::{to_repr}

///|
pub extend DecoderFoundationCudaOperation with Eq::{not_equal, equal}

///|
pub extend DecoderFoundationNumericalContract with @debug.Debug::{to_repr}

///|
pub extend DecoderFoundationNumericalContract with Eq::{not_equal, equal}

///|
pub extend DecoderFoundationCudaSourceError with @debug.Debug::{to_repr}

///|
pub extend DecoderFoundationCudaSourceError with Eq::{not_equal, equal}
