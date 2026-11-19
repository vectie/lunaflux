///|
pub extend RuntimeCapacityRule with @debug.Debug::{to_repr}

///|
pub extend RuntimeCapacityRule with Eq::{not_equal, equal}

///|
pub extend RuntimeCapacityError with @debug.Debug::{to_repr}

///|
pub extend RuntimeCapacityError with Eq::{not_equal, equal}

///|
pub extend RuntimeCapacitySubject with @debug.Debug::{to_repr}

///|
pub extend RuntimeCapacitySubject with Eq::{not_equal, equal}

///|
pub extend RuntimeCapacityReason with @debug.Debug::{to_repr}

///|
pub extend RuntimeCapacityReason with Eq::{not_equal, equal}

///|
pub extend RuntimeCapacityDecision with @debug.Debug::{to_repr}

///|
pub extend RuntimeCapacityDecision with Eq::{not_equal, equal}

///|
pub extend ResolvedRuntimeCapacity with @debug.Debug::{to_repr}

///|
pub extend ResolvedRuntimeCapacity with Eq::{not_equal, equal}

///|
pub extend RuntimeGraphMemoryAvailability with @debug.Debug::{to_repr}

///|
pub extend RuntimeGraphMemoryAvailability with Eq::{not_equal, equal}

///|
pub extend ResolvedGraphMemoryCapacity with @debug.Debug::{to_repr}

///|
pub extend ResolvedGraphMemoryCapacity with Eq::{not_equal, equal}
