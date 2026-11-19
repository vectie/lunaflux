///|
pub extend LunaBf16KernelProducerRule with @debug.Debug::{to_repr}

///|
pub extend LunaBf16KernelProducerRule with Eq::{not_equal, equal}

///|
pub extend LunaBf16KernelProducerError with @debug.Debug::{to_repr}

///|
pub extend LunaBf16KernelProducerError with Eq::{not_equal, equal}

///|
pub extend LunaBf16PromotionRule with @debug.Debug::{to_repr}

///|
pub extend LunaBf16PromotionRule with Eq::{not_equal, equal}

///|
pub extend LunaBf16PromotionError with @debug.Debug::{to_repr}

///|
pub extend LunaBf16PromotionError with Eq::{not_equal, equal}

///|
pub extend LunaBf16PromotionDigest with @debug.Debug::{to_repr}

///|
pub extend LunaBf16PromotionDigest with Eq::{not_equal, equal}

///|
pub extend LunaBf16PromotionDigest with Hash::{hash, hash_combine}

///|
pub extend LunaBf16BenchmarkShape with @debug.Debug::{to_repr}

///|
pub extend LunaBf16BenchmarkShape with Eq::{not_equal, equal}

///|
pub extend LunaBf16KernelCandidate with @debug.Debug::{to_repr}

///|
pub extend LunaBf16KernelCandidate with Eq::{not_equal, equal}
