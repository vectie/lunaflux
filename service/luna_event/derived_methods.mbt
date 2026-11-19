///|
pub extend LunaEventRule with @debug.Debug::{to_repr}

///|
pub extend LunaEventRule with Eq::{not_equal, equal}

///|
pub extend LunaEventError with @debug.Debug::{to_repr}

///|
pub extend LunaEventError with Eq::{not_equal, equal}

///|
pub extend LunaEventPublishStatus with @debug.Debug::{to_repr}

///|
pub extend LunaEventPublishStatus with Eq::{not_equal, equal}

///|
pub extend LunaEventKind with @debug.Debug::{to_repr}

///|
pub extend LunaEventKind with Eq::{not_equal, equal}
