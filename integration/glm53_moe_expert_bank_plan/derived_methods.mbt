///|
pub extend Glm53MoeExpertBankPlanError with @debug.Debug::{to_repr}

///|
pub extend Glm53MoeExpertBankPlanError with Eq::{not_equal, equal}

///|
pub extend Glm53MoeExpertBankCandidatePlanDigest with @debug.Debug::{to_repr}

///|
pub extend Glm53MoeExpertBankCandidatePlanDigest with Eq::{not_equal, equal}

///|
pub extend Glm53MoeExpertBankCandidatePlanDigest with Hash::{hash, hash_combine}
