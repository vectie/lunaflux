///|
pub extend Fp8RuntimeRecipeRule with @debug.Debug::{to_repr}

///|
pub extend Fp8RuntimeRecipeRule with Eq::{not_equal, equal}

///|
pub extend Fp8RuntimeRecipeError with @debug.Debug::{to_repr}

///|
pub extend Fp8RuntimeRecipeError with Eq::{not_equal, equal}

///|
pub extend Fp8RuntimeRecipeLimits with @debug.Debug::{to_repr}

///|
pub extend Fp8RuntimeRecipeLimits with Eq::{not_equal, equal}

///|
pub extend Fp8RuntimeRecipeDigest with @debug.Debug::{to_repr}

///|
pub extend Fp8RuntimeRecipeDigest with Eq::{not_equal, equal}

///|
pub extend Fp8RuntimeRecipeDigest with Hash::{hash, hash_combine}

///|
pub extend Fp8WeightOperandStorageRegion with @debug.Debug::{to_repr}

///|
pub extend Fp8WeightOperandStorageRegion with Eq::{not_equal, equal}

///|
pub extend Fp8RuntimeRecipeV2Rule with @debug.Debug::{to_repr}

///|
pub extend Fp8RuntimeRecipeV2Rule with Eq::{not_equal, equal}

///|
pub extend Fp8RuntimeRecipeV2Error with @debug.Debug::{to_repr}

///|
pub extend Fp8RuntimeRecipeV2Error with Eq::{not_equal, equal}

///|
pub extend Fp8RuntimeRecipeV2Limits with @debug.Debug::{to_repr}

///|
pub extend Fp8RuntimeRecipeV2Limits with Eq::{not_equal, equal}

///|
pub extend Fp8RuntimeRecipeV2Digest with @debug.Debug::{to_repr}

///|
pub extend Fp8RuntimeRecipeV2Digest with Eq::{not_equal, equal}

///|
pub extend Fp8RuntimeRecipeV2Digest with Hash::{hash, hash_combine}

///|
pub extend Fp8ReusablePagedRecipeV3Rule with @debug.Debug::{to_repr}

///|
pub extend Fp8ReusablePagedRecipeV3Rule with Eq::{not_equal, equal}

///|
pub extend Fp8ReusablePagedRecipeV3Error with @debug.Debug::{to_repr}

///|
pub extend Fp8ReusablePagedRecipeV3Error with Eq::{not_equal, equal}

///|
pub extend Fp8ReusablePagedRecipeV3Digest with @debug.Debug::{to_repr}

///|
pub extend Fp8ReusablePagedRecipeV3Digest with Eq::{not_equal, equal}

///|
pub extend Fp8ReusablePagedRecipeV3Digest with Hash::{hash, hash_combine}
