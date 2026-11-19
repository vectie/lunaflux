///|
pub extend GlmTinyFeedForwardPlan with @debug.Debug::{to_repr}

///|
pub extend GlmTinyFeedForwardPlan with Eq::{not_equal, equal}

///|
pub extend GlmTinyCompositionPlan with @debug.Debug::{to_repr}

///|
pub extend GlmTinyCompositionPlan with Eq::{not_equal, equal}

///|
pub extend GlmHybridReferenceError with @debug.Debug::{to_repr}

///|
pub extend GlmHybridReferenceError with Eq::{not_equal, equal}

///|
pub extend GlmReferenceAttentionStep with @debug.Debug::{to_repr}

///|
pub extend GlmReferenceAttentionStep with Eq::{not_equal, equal}

///|
pub extend GlmReferenceFeedForwardStep with @debug.Debug::{to_repr}

///|
pub extend GlmReferenceFeedForwardStep with Eq::{not_equal, equal}
