///|
pub extend Qwen35ConfigArea with @debug.Debug::{to_repr}

///|
pub extend Qwen35ConfigArea with Eq::{not_equal, equal}

///|
pub extend Qwen35ConfigField with @debug.Debug::{to_repr}

///|
pub extend Qwen35ConfigField with Eq::{not_equal, equal}

///|
pub extend Qwen35ConfigValueIssue with @debug.Debug::{to_repr}

///|
pub extend Qwen35ConfigValueIssue with Eq::{not_equal, equal}

///|
pub extend Qwen35ConfigError with @debug.Debug::{to_repr}

///|
pub extend Qwen35ConfigError with Eq::{not_equal, equal}
