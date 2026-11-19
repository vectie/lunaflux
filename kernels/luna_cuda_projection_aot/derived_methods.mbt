///|
pub extend ProjectionCudaSourceDigest with @debug.Debug::{to_repr}

///|
pub extend ProjectionCudaSourceDigest with Eq::{not_equal, equal}

///|
pub extend ProjectionCudaSourceDigest with Hash::{hash, hash_combine}

///|
pub extend ProjectionCudaRecipeDigest with @debug.Debug::{to_repr}

///|
pub extend ProjectionCudaRecipeDigest with Eq::{not_equal, equal}

///|
pub extend ProjectionCudaRecipeDigest with Hash::{hash, hash_combine}

///|
pub extend ProjectionCudaManifestBindingDigest with @debug.Debug::{to_repr}

///|
pub extend ProjectionCudaManifestBindingDigest with Eq::{not_equal, equal}

///|
pub extend ProjectionCudaManifestBindingDigest with Hash::{hash, hash_combine}

///|
pub extend ProjectionAotRule with @debug.Debug::{to_repr}

///|
pub extend ProjectionAotRule with Eq::{not_equal, equal}

///|
pub extend ProjectionAotError with @debug.Debug::{to_repr}

///|
pub extend ProjectionAotError with Eq::{not_equal, equal}

///|
pub extend ProjectionAotFamily with @debug.Debug::{to_repr}

///|
pub extend ProjectionAotFamily with Eq::{not_equal, equal}

///|
pub extend ProjectionNumericalContract with @debug.Debug::{to_repr}

///|
pub extend ProjectionNumericalContract with Eq::{not_equal, equal}

///|
pub extend ProjectionAotLimits with @debug.Debug::{to_repr}

///|
pub extend ProjectionAotLimits with Eq::{not_equal, equal}

///|
pub extend ProjectionCudaLowering with @debug.Debug::{to_repr}

///|
pub extend ProjectionCudaLowering with Eq::{not_equal, equal}

///|
pub extend ProjectionManifestCudaLowering with @debug.Debug::{to_repr}

///|
pub extend ProjectionManifestCudaLowering with Eq::{not_equal, equal}
