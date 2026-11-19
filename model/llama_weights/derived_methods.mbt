///|
pub extend LlamaWeightsError with @debug.Debug::{to_repr}

///|
pub extend LlamaWeightsError with Eq::{not_equal, equal}

///|
pub extend TensorBinding with @debug.Debug::{to_repr}

///|
pub extend TensorBinding with Eq::{not_equal, equal}

///|
pub extend LlamaWeightBindings with @debug.Debug::{to_repr}

///|
pub extend LlamaWeightBindings with Eq::{not_equal, equal}
