///|
pub extend PointwiseSourceDigest with @debug.Debug::{to_repr}

///|
pub extend PointwiseSourceDigest with Eq::{not_equal, equal}

///|
pub extend PointwiseSourceDigest with Hash::{hash, hash_combine}

///|
pub extend PointwiseRecipeDigest with @debug.Debug::{to_repr}

///|
pub extend PointwiseRecipeDigest with Eq::{not_equal, equal}

///|
pub extend PointwiseRecipeDigest with Hash::{hash, hash_combine}

///|
pub extend PointwiseManifestBindingDigest with @debug.Debug::{to_repr}

///|
pub extend PointwiseManifestBindingDigest with Eq::{not_equal, equal}

///|
pub extend PointwiseManifestBindingDigest with Hash::{hash, hash_combine}

///|
pub extend PointwiseAotRule with @debug.Debug::{to_repr}

///|
pub extend PointwiseAotRule with Eq::{not_equal, equal}

///|
pub extend PointwiseAotError with @debug.Debug::{to_repr}

///|
pub extend PointwiseAotError with Eq::{not_equal, equal}

///|
pub extend PointwiseAotFamily with @debug.Debug::{to_repr}

///|
pub extend PointwiseAotFamily with Eq::{not_equal, equal}

///|
pub extend PointwiseNumericalContract with @debug.Debug::{to_repr}

///|
pub extend PointwiseNumericalContract with Eq::{not_equal, equal}

///|
pub extend PointwiseAotLimits with @debug.Debug::{to_repr}

///|
pub extend PointwiseAotLimits with Eq::{not_equal, equal}

///|
pub extend PointwiseCudaLowering with @debug.Debug::{to_repr}

///|
pub extend PointwiseCudaLowering with Eq::{not_equal, equal}

///|
pub extend PointwiseManifestCudaLowering with @debug.Debug::{to_repr}

///|
pub extend PointwiseManifestCudaLowering with Eq::{not_equal, equal}
