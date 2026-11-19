///|
pub extend ModelFamilyId with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ModelFamilyId with Eq::{not_equal, equal}

///|
pub extend ModelWorkload with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ModelWorkload with Eq::{not_equal, equal}

///|
pub extend ModelSemanticCapability with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ModelSemanticCapability with Eq::{not_equal, equal}

///|
pub extend FamilyContractError with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend FamilyContractError with Eq::{not_equal, equal}

///|
pub extend ModelFamilyRequirements with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ModelFamilyRequirements with Eq::{not_equal, equal}

///|
pub extend ModelBackendCapabilities with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend ModelBackendCapabilities with Eq::{not_equal, equal}

///|
pub extend ModelFamilyAdmission with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ModelFamilyAdmission with Eq::{not_equal, equal}
