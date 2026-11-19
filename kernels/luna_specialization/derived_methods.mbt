///|
pub extend LunaSpecializationRule with @debug.Debug::{to_repr}

///|
pub extend LunaSpecializationRule with Eq::{not_equal, equal}

///|
pub extend LunaSpecializationError with @debug.Debug::{to_repr}

///|
pub extend LunaSpecializationError with Eq::{not_equal, equal}

///|
pub extend LunaCompilerIdentity with @debug.Debug::{to_repr}

///|
pub extend LunaCompilerIdentity with Eq::{not_equal, equal}

///|
pub extend LunaCompilerPolicy with @debug.Debug::{to_repr}

///|
pub extend LunaCompilerPolicy with Eq::{not_equal, equal}

///|
pub extend LunaNumericalPolicy with @debug.Debug::{to_repr}

///|
pub extend LunaNumericalPolicy with Eq::{not_equal, equal}

///|
pub extend LunaQuantizationPolicy with @debug.Debug::{to_repr}

///|
pub extend LunaQuantizationPolicy with Eq::{not_equal, equal}

///|
pub extend LunaSpecializationEvidenceDigest with @debug.Debug::{to_repr}

///|
pub extend LunaSpecializationEvidenceDigest with Eq::{not_equal, equal}

///|
pub extend LunaSpecializationEvidenceDigest with Hash::{hash, hash_combine}

///|
pub extend LunaSpecializationEvidence with @debug.Debug::{to_repr}

///|
pub extend LunaSpecializationEvidence with Eq::{not_equal, equal}

///|
pub extend LunaSpecializationLimits with @debug.Debug::{to_repr}

///|
pub extend LunaSpecializationLimits with Eq::{not_equal, equal}

///|
pub extend LunaSpecializedEntryPointId with @debug.Debug::{to_repr}

///|
pub extend LunaSpecializedEntryPointId with Eq::{not_equal, equal}

///|
pub extend LunaSpecializedEntryPointId with Hash::{hash, hash_combine}

///|
pub extend LunaSpecializedRegionSpace with @debug.Debug::{to_repr}

///|
pub extend LunaSpecializedRegionSpace with Eq::{not_equal, equal}

///|
pub extend LunaSpecializedRegion with @debug.Debug::{to_repr}

///|
pub extend LunaSpecializedRegion with Eq::{not_equal, equal}

///|
pub extend LunaSpecializationDigest with @debug.Debug::{to_repr}

///|
pub extend LunaSpecializationDigest with Eq::{not_equal, equal}

///|
pub extend LunaSpecializationDigest with Hash::{hash, hash_combine}

///|
pub extend LunaSpecializationRecord with @debug.Debug::{to_repr}

///|
pub extend LunaSpecializationRecord with Eq::{not_equal, equal}
