///|
pub extend OpenAIResponsesCampaignError with @debug.Debug::{to_repr}

///|
pub extend OpenAIResponsesCampaignError with Eq::{not_equal, equal}
