///|
pub extend NumericWeightFileError with @debug.Debug::{to_repr}

///|
pub extend NumericWeightFileError with Eq::{not_equal, equal}

///|
pub extend NumericWeightAuthorityRule with @debug.Debug::{to_repr}

///|
pub extend NumericWeightAuthorityRule with Eq::{not_equal, equal}

///|
pub extend NumericWeightFileLimit with @debug.Debug::{to_repr}

///|
pub extend NumericWeightFileLimit with Eq::{not_equal, equal}

///|
pub extend NumericWeightFailureStage with @debug.Debug::{to_repr}

///|
pub extend NumericWeightFailureStage with Eq::{not_equal, equal}

///|
pub extend NumericWeightArtifactDigest with @debug.Debug::{to_repr}

///|
pub extend NumericWeightArtifactDigest with Eq::{not_equal, equal}

///|
pub extend NumericWeightArtifactDigest with Hash::{hash, hash_combine}
