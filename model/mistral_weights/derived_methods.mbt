///|
pub extend MistralWeightRule with @debug.Debug::{to_repr}

///|
pub extend MistralWeightRule with Eq::{not_equal, equal}

///|
pub extend MistralWeightsError with @debug.Debug::{to_repr}

///|
pub extend MistralWeightsError with Eq::{not_equal, equal}

///|
pub extend MistralWeightBinding with @debug.Debug::{to_repr}

///|
pub extend MistralWeightBinding with Eq::{not_equal, equal}

///|
pub extend MistralWeightManifest with @debug.Debug::{to_repr}

///|
pub extend MistralWeightManifest with Eq::{not_equal, equal}
