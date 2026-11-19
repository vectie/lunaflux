///|
pub extend LunaAotAdmissionRule with @debug.Debug::{to_repr}

///|
pub extend LunaAotAdmissionRule with Eq::{not_equal, equal}

///|
pub extend LunaAotAdmissionError with @debug.Debug::{to_repr}

///|
pub extend LunaAotAdmissionError with Eq::{not_equal, equal}

///|
pub extend LunaAotKernelAdmission with @debug.Debug::{to_repr}

///|
pub extend LunaAotKernelAdmission with Eq::{not_equal, equal}
