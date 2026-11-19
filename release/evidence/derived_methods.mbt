///|
pub extend ReleaseEvidenceArtifact with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ReleaseEvidenceArtifact with Eq::{not_equal, equal}

///|
pub extend ReleaseEvidenceEntry with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ReleaseEvidenceEntry with Eq::{not_equal, equal}

///|
pub extend ReleaseReviewerEntry with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ReleaseReviewerEntry with Eq::{not_equal, equal}

///|
pub extend ReleaseEvidenceStatus with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ReleaseEvidenceStatus with Eq::{not_equal, equal}

///|
pub extend ReleaseEvidenceRole with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ReleaseEvidenceRole with Eq::{not_equal, equal}

///|
pub extend ReleaseReviewerRole with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ReleaseReviewerRole with Eq::{not_equal, equal}

///|
pub extend ReleaseEvidenceError with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ReleaseEvidenceError with Eq::{not_equal, equal}

///|
pub extend ReleaseCampaignDigest with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ReleaseCampaignDigest with Eq::{not_equal, equal}

///|
pub extend ReleaseCampaignDigest with Hash::{hash, hash_combine}

///|
pub extend ReleaseSubjectDigest with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ReleaseSubjectDigest with Eq::{not_equal, equal}

///|
pub extend ReleaseSubjectDigest with Hash::{hash, hash_combine}

///|
pub extend ReleaseImageDigest with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ReleaseImageDigest with Eq::{not_equal, equal}

///|
pub extend ReleaseImageDigest with Hash::{hash, hash_combine}

///|
pub extend ReleaseTargetDigest with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ReleaseTargetDigest with Eq::{not_equal, equal}

///|
pub extend ReleaseTargetDigest with Hash::{hash, hash_combine}

///|
pub extend ReleaseArtifactDigest with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ReleaseArtifactDigest with Eq::{not_equal, equal}

///|
pub extend ReleaseArtifactDigest with Hash::{hash, hash_combine}

///|
pub extend ReleaseReviewerIdentityDigest with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend ReleaseReviewerIdentityDigest with Eq::{not_equal, equal}

///|
pub extend ReleaseReviewerIdentityDigest with Hash::{hash, hash_combine}

///|
pub extend ReleaseCampaignRecord with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ReleaseCampaignRecord with Eq::{not_equal, equal}

///|
pub extend ReleaseEvidenceObservation with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend ReleaseEvidenceObservation with Eq::{not_equal, equal}

///|
pub extend ReleaseReviewerObservation with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend ReleaseReviewerObservation with Eq::{not_equal, equal}

///|
pub extend VerifiedReleaseCampaign with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend VerifiedReleaseCampaign with Eq::{not_equal, equal}
