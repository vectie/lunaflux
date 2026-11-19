///|
pub extend JointConditioningAotGap with @debug.Debug::{to_repr}

///|
pub extend JointConditioningAotGap with Eq::{not_equal, equal}

///|
pub extend JointConditioningReferenceError with @debug.Debug::{to_repr}

///|
pub extend JointConditioningReferenceError with Eq::{not_equal, equal}

///|
pub extend JointConditioningAotGapDigest with @debug.Debug::{to_repr}

///|
pub extend JointConditioningAotGapDigest with Eq::{not_equal, equal}

///|
pub extend JointConditioningAotGapDigest with Hash::{hash, hash_combine}
