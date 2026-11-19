///|
pub extend Fp8ProjectionAotRule with @debug.Debug::{to_repr}

///|
pub extend Fp8ProjectionAotRule with Eq::{not_equal, equal}

///|
pub extend Fp8ProjectionAotError with @debug.Debug::{to_repr}

///|
pub extend Fp8ProjectionAotError with Eq::{not_equal, equal}

///|
pub extend Fp8ProjectionFamily with @debug.Debug::{to_repr}

///|
pub extend Fp8ProjectionFamily with Eq::{not_equal, equal}

///|
pub extend Fp8ProjectionNumericalContract with @debug.Debug::{to_repr}

///|
pub extend Fp8ProjectionNumericalContract with Eq::{not_equal, equal}

///|
pub extend Fp8ProjectionCudaSourceDigest with @debug.Debug::{to_repr}

///|
pub extend Fp8ProjectionCudaSourceDigest with Eq::{not_equal, equal}

///|
pub extend Fp8ProjectionCudaSourceDigest with Hash::{hash, hash_combine}

///|
pub extend Fp8ProjectionCudaRecipeDigest with @debug.Debug::{to_repr}

///|
pub extend Fp8ProjectionCudaRecipeDigest with Eq::{not_equal, equal}

///|
pub extend Fp8ProjectionCudaRecipeDigest with Hash::{hash, hash_combine}

///|
pub extend Fp8ProjectionManifestBindingDigest with @debug.Debug::{to_repr}

///|
pub extend Fp8ProjectionManifestBindingDigest with Eq::{not_equal, equal}

///|
pub extend Fp8ProjectionManifestBindingDigest with Hash::{hash, hash_combine}

///|
pub extend Fp8ProjectionCudaLowering with @debug.Debug::{to_repr}

///|
pub extend Fp8ProjectionCudaLowering with Eq::{not_equal, equal}

///|
pub extend Fp8ProjectionManifestCudaLowering with @debug.Debug::{to_repr}

///|
pub extend Fp8ProjectionManifestCudaLowering with Eq::{not_equal, equal}

///|
pub extend Fp8ProjectionV2NumericalContract with @debug.Debug::{to_repr}

///|
pub extend Fp8ProjectionV2NumericalContract with Eq::{not_equal, equal}

///|
pub extend Fp8ProjectionCudaLoweringV2 with @debug.Debug::{to_repr}

///|
pub extend Fp8ProjectionCudaLoweringV2 with Eq::{not_equal, equal}

///|
pub extend Fp8ProjectionManifestCudaLoweringV2 with @debug.Debug::{to_repr}

///|
pub extend Fp8ProjectionManifestCudaLoweringV2 with Eq::{not_equal, equal}
