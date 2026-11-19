///|
pub extend JointTinyConditioningSource with @debug.Debug::{to_repr}

///|
pub extend JointTinyConditioningSource with Eq::{not_equal, equal}

///|
pub extend JointTinyTransformerReferenceError with @debug.Debug::{to_repr}

///|
pub extend JointTinyTransformerReferenceError with Eq::{not_equal, equal}

///|
pub extend JointTinyConditioningMatrix with @debug.Debug::{to_repr}

///|
pub extend JointTinyConditioningMatrix with Eq::{not_equal, equal}

///|
pub extend JointTinyConditioningInput with @debug.Debug::{to_repr}

///|
pub extend JointTinyConditioningInput with Eq::{not_equal, equal}

///|
pub extend JointTinyDense with @debug.Debug::{to_repr}

///|
pub extend JointTinyDense with Eq::{not_equal, equal}

///|
pub extend JointTinyMlp with @debug.Debug::{to_repr}

///|
pub extend JointTinyMlp with Eq::{not_equal, equal}
