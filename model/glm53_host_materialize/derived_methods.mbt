///|
pub extend Glm53HostLimit with @debug.Debug::{to_repr}

///|
pub extend Glm53HostLimit with Eq::{not_equal, equal}

///|
pub extend Glm53HostDigestKind with @debug.Debug::{to_repr}

///|
pub extend Glm53HostDigestKind with Eq::{not_equal, equal}

///|
pub extend Glm53HostMaterializeError with @debug.Debug::{to_repr}

///|
pub extend Glm53HostMaterializeError with Eq::{not_equal, equal}

///|
pub extend Glm53HostSourceDigest with @debug.Debug::{to_repr}

///|
pub extend Glm53HostSourceDigest with Eq::{not_equal, equal}

///|
pub extend Glm53HostSourceDigest with Hash::{hash, hash_combine}
