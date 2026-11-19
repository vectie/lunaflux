///|
pub extend LunaCudaSourceDigest with @debug.Debug::{to_repr}

///|
pub extend LunaCudaSourceDigest with Eq::{not_equal, equal}

///|
pub extend LunaCudaSourceDigest with Hash::{hash, hash_combine}

///|
pub extend LunaCudaRecipeDigest with @debug.Debug::{to_repr}

///|
pub extend LunaCudaRecipeDigest with Eq::{not_equal, equal}

///|
pub extend LunaCudaRecipeDigest with Hash::{hash, hash_combine}

///|
pub extend LunaCudaArtifactDigest with @debug.Debug::{to_repr}

///|
pub extend LunaCudaArtifactDigest with Eq::{not_equal, equal}

///|
pub extend LunaCudaArtifactDigest with Hash::{hash, hash_combine}

///|
pub extend LunaCudaEvidenceDigest with @debug.Debug::{to_repr}

///|
pub extend LunaCudaEvidenceDigest with Eq::{not_equal, equal}

///|
pub extend LunaCudaEvidenceDigest with Hash::{hash, hash_combine}

///|
pub extend LunaCudaAotRule with @debug.Debug::{to_repr}

///|
pub extend LunaCudaAotRule with Eq::{not_equal, equal}

///|
pub extend LunaCudaAotError with @debug.Debug::{to_repr}

///|
pub extend LunaCudaAotError with Eq::{not_equal, equal}

///|
pub extend LunaCudaAotLimits with @debug.Debug::{to_repr}

///|
pub extend LunaCudaAotLimits with Eq::{not_equal, equal}

///|
pub extend LunaCudaAotLowering with @debug.Debug::{to_repr}

///|
pub extend LunaCudaAotLowering with Eq::{not_equal, equal}

///|
pub extend LunaCudaOfflineCompileReceipt with @debug.Debug::{to_repr}

///|
pub extend LunaCudaOfflineCompileReceipt with Eq::{not_equal, equal}

///|
pub extend LunaCudaAotArtifact with @debug.Debug::{to_repr}

///|
pub extend LunaCudaAotArtifact with Eq::{not_equal, equal}

///|
pub extend LunaCudaAotSpecializedArtifact with @debug.Debug::{to_repr}

///|
pub extend LunaCudaAotSpecializedArtifact with Eq::{not_equal, equal}
