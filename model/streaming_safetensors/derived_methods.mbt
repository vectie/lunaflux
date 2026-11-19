///|
pub extend StreamingSafetensorsDTypeKind with @debug.Debug::{to_repr}

///|
pub extend StreamingSafetensorsDTypeKind with Eq::{not_equal, equal}

///|
pub extend StreamingSafetensorsDTypeSpec with @debug.Debug::{to_repr}

///|
pub extend StreamingSafetensorsDTypeSpec with Eq::{not_equal, equal}

///|
pub extend StreamingSafetensorsFileDigest with @debug.Debug::{to_repr}

///|
pub extend StreamingSafetensorsFileDigest with Eq::{not_equal, equal}

///|
pub extend StreamingSafetensorsFileDigest with Hash::{hash, hash_combine}

///|
pub extend StreamingSafetensorsLimits with @debug.Debug::{to_repr}

///|
pub extend StreamingSafetensorsLimits with Eq::{not_equal, equal}

///|
pub extend StreamingSafetensorsTensorMetadata with @debug.Debug::{to_repr}

///|
pub extend StreamingSafetensorsTensorMetadata with Eq::{not_equal, equal}

///|
pub extend StreamingSafetensorsShardMetadata with @debug.Debug::{to_repr}

///|
pub extend StreamingSafetensorsShardMetadata with Eq::{not_equal, equal}

///|
pub extend StreamingSafetensorsCopyRequest with @debug.Debug::{to_repr}

///|
pub extend StreamingSafetensorsCopyRequest with Eq::{not_equal, equal}

///|
pub extend StreamingSafetensorsVisitedRange with @debug.Debug::{to_repr}

///|
pub extend StreamingSafetensorsVisitedRange with Eq::{not_equal, equal}

///|
pub extend StreamingSafetensorsSegmentedCopyRequest with @debug.Debug::{to_repr}

///|
pub extend StreamingSafetensorsSegmentedCopyRequest with Eq::{not_equal, equal}

///|
pub extend StreamingSafetensorsSegmentedVisitedRange with @debug.Debug::{
  to_repr,
}

///|
pub extend StreamingSafetensorsSegmentedVisitedRange with Eq::{not_equal, equal}

///|
pub extend StreamingSafetensorsJsonField with @debug.Debug::{to_repr}

///|
pub extend StreamingSafetensorsJsonField with Eq::{not_equal, equal}

///|
pub extend StreamingSafetensorsError with @debug.Debug::{to_repr}

///|
pub extend StreamingSafetensorsError with Eq::{not_equal, equal}
