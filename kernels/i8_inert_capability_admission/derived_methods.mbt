///|
pub extend I8InertKernelCapabilityAdmissionVersion with @debug.Debug::{to_repr}

///|
pub extend I8InertKernelCapabilityAdmissionVersion with Eq::{not_equal, equal}

///|
pub extend I8InertKernelCapabilityAdmissionDigest with @debug.Debug::{to_repr}

///|
pub extend I8InertKernelCapabilityAdmissionDigest with Eq::{not_equal, equal}

///|
pub extend I8InertKernelCapabilityAdmissionDigest with Hash::{
  hash,
  hash_combine,
}

///|
pub extend I8InertKernelCapabilityAdmissionLimits with @debug.Debug::{to_repr}

///|
pub extend I8InertKernelCapabilityAdmissionLimits with Eq::{not_equal, equal}

///|
pub extend I8InertKernelCapabilityAdmissionRule with @debug.Debug::{to_repr}

///|
pub extend I8InertKernelCapabilityAdmissionRule with Eq::{not_equal, equal}

///|
pub extend I8InertKernelCapabilityAdmissionError with @debug.Debug::{to_repr}

///|
pub extend I8InertKernelCapabilityAdmissionError with Eq::{not_equal, equal}

///|
pub extend I8InertWeightEvidence with @debug.Debug::{to_repr}

///|
pub extend I8InertWeightEvidence with Eq::{not_equal, equal}

///|
pub extend I8InertOperationEvidence with @debug.Debug::{to_repr}

///|
pub extend I8InertOperationEvidence with Eq::{not_equal, equal}

///|
pub extend I8InertKernelCapabilityAdmission with @debug.Debug::{to_repr}

///|
pub extend I8InertKernelCapabilityAdmission with Eq::{not_equal, equal}

///|
pub extend I8ProductionPolicyAdmission with @debug.Debug::{to_repr}

///|
pub extend I8ProductionPolicyAdmission with Eq::{not_equal, equal}
