///|
pub extend OpenAIBaselineAdapterError with @debug.Debug::{to_repr}

///|
pub extend OpenAIBaselineAdapterError with Eq::{not_equal, equal}

///|
pub extend OpenAIBaselineObservedProtocol with @debug.Debug::{to_repr}

///|
pub extend OpenAIBaselineObservedProtocol with Eq::{not_equal, equal}
