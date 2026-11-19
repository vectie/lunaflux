///|
pub extend DeepSeekV4ConfigError with @debug.Debug::{to_repr}

///|
pub extend DeepSeekV4ConfigError with Eq::{not_equal, equal}
