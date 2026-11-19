///|
pub extend PrefixLimitRule with @debug.Debug::{to_repr}

///|
pub extend PrefixLimitRule with Eq::{not_equal, equal}

///|
pub extend PrefixInvariantRule with @debug.Debug::{to_repr}

///|
pub extend PrefixInvariantRule with Eq::{not_equal, equal}

///|
pub extend PrefixIndexError with @debug.Debug::{to_repr}

///|
pub extend PrefixIndexError with Eq::{not_equal, equal}

///|
pub extend PrefixEntryId with @debug.Debug::{to_repr}

///|
pub extend PrefixEntryId with Eq::{not_equal, equal}

///|
pub extend PrefixEntryId with Hash::{hash, hash_combine}

///|
pub extend PrefixLayout with @debug.Debug::{to_repr}

///|
pub extend PrefixLayout with Eq::{not_equal, equal}

///|
pub extend PrefixIndexLimits with @debug.Debug::{to_repr}

///|
pub extend PrefixIndexLimits with Eq::{not_equal, equal}

///|
pub extend PrefixEntrySnapshot with @debug.Debug::{to_repr}

///|
pub extend PrefixEntrySnapshot with Eq::{not_equal, equal}

///|
pub extend PrefixIndexSnapshot with @debug.Debug::{to_repr}

///|
pub extend PrefixIndexSnapshot with Eq::{not_equal, equal}
