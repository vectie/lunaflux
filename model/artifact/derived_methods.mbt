///|
pub extend ArtifactFile with @debug.Debug::{to_repr}

///|
pub extend ArtifactFile with Eq::{not_equal, equal}

///|
pub extend ArtifactFileOperation with @debug.Debug::{to_repr}

///|
pub extend ArtifactFileOperation with Eq::{not_equal, equal}

///|
pub extend ArtifactFileIoIssue with @debug.Debug::{to_repr}

///|
pub extend ArtifactFileIoIssue with Eq::{not_equal, equal}

///|
pub extend ArtifactLimit with @debug.Debug::{to_repr}

///|
pub extend ArtifactLimit with Eq::{not_equal, equal}

///|
pub extend ArtifactError with @debug.Debug::{to_repr}

///|
pub extend ArtifactError with Eq::{not_equal, equal}

///|
pub extend ArtifactDigest with @debug.Debug::{to_repr}

///|
pub extend ArtifactDigest with Eq::{not_equal, equal}

///|
pub extend ExpectedDigests with @debug.Debug::{to_repr}

///|
pub extend ExpectedDigests with Eq::{not_equal, equal}

///|
pub extend ArtifactLimits with @debug.Debug::{to_repr}

///|
pub extend ArtifactLimits with Eq::{not_equal, equal}

///|
pub extend ArtifactIdentity with @debug.Debug::{to_repr}

///|
pub extend ArtifactIdentity with Eq::{not_equal, equal}
