///|
pub extend LunaRequestPreparationAdmissionKind with @debug.Debug::{to_repr}

///|
pub extend LunaRequestPreparationAdmissionKind with Eq::{not_equal, equal}

///|
pub extend LunaRequestPreparationWorkState with @debug.Debug::{to_repr}

///|
pub extend LunaRequestPreparationWorkState with Eq::{not_equal, equal}

///|
pub extend LunaRequestPreparationPoolProgress with @debug.Debug::{to_repr}

///|
pub extend LunaRequestPreparationPoolProgress with Eq::{not_equal, equal}

///|
pub extend RequestAdmissionRule with @debug.Debug::{to_repr}

///|
pub extend RequestAdmissionRule with Eq::{not_equal, equal}

///|
pub extend RequestAdmissionError with @debug.Debug::{to_repr}

///|
pub extend RequestAdmissionError with Eq::{not_equal, equal}
