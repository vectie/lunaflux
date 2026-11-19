///|
pub extend LunaTileOptimizerPromotionRule with @debug.Debug::{to_repr}

///|
pub extend LunaTileOptimizerPromotionRule with Eq::{not_equal, equal}

///|
pub extend LunaTileOptimizerPromotionError with @debug.Debug::{to_repr}

///|
pub extend LunaTileOptimizerPromotionError with Eq::{not_equal, equal}

///|
pub extend LunaTileOptimizerDigest with @debug.Debug::{to_repr}

///|
pub extend LunaTileOptimizerDigest with Eq::{not_equal, equal}

///|
pub extend LunaTileOptimizerDigest with Hash::{hash, hash_combine}

///|
pub extend LunaTileOptimizerPath with @debug.Debug::{to_repr}

///|
pub extend LunaTileOptimizerPath with Eq::{not_equal, equal}

///|
pub extend LunaTilePromotionDeploymentIdentity with @debug.Debug::{to_repr}

///|
pub extend LunaTilePromotionDeploymentIdentity with Eq::{not_equal, equal}

///|
pub extend LunaTilePairedBenchmarkObservation with @debug.Debug::{to_repr}

///|
pub extend LunaTilePairedBenchmarkObservation with Eq::{not_equal, equal}
