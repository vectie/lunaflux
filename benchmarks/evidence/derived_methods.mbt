///|
pub extend BenchmarkDigest with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend BenchmarkDigest with Eq::{not_equal, equal}

///|
pub extend BenchmarkDigest with Hash::{hash, hash_combine}

///|
pub extend BenchmarkEngine with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend BenchmarkEngine with Eq::{not_equal, equal}

///|
pub extend BenchmarkProfile with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend BenchmarkProfile with Eq::{not_equal, equal}

///|
pub extend BenchmarkRequestOutcome with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend BenchmarkRequestOutcome with Eq::{not_equal, equal}

///|
pub extend BenchmarkEvidenceError with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend BenchmarkEvidenceError with Eq::{not_equal, equal}
