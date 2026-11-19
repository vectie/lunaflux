///|
pub extend Qwen3WeightRule with @debug.Debug::{to_repr}

///|
pub extend Qwen3WeightRule with Eq::{not_equal, equal}

///|
pub extend Qwen3WeightsError with @debug.Debug::{to_repr}

///|
pub extend Qwen3WeightsError with Eq::{not_equal, equal}

///|
pub extend Qwen3TensorBinding with @debug.Debug::{to_repr}

///|
pub extend Qwen3TensorBinding with Eq::{not_equal, equal}

///|
pub extend Qwen3WeightBindings with @debug.Debug::{to_repr}

///|
pub extend Qwen3WeightBindings with Eq::{not_equal, equal}

///|
pub extend Qwen3SourceFileLimits with @debug.Debug::{to_repr}

///|
pub extend Qwen3SourceFileLimits with Eq::{not_equal, equal}
