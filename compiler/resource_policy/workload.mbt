///|
/// Offline timing objective. Actual work and captured launch bucket are both
/// part of identity: a maximum-envelope probe is not a small-bucket sample.
pub struct Workload {
  priv phase : String
  priv queries : Int
  priv rows : Int
  priv history : Int
  priv bucket_queries : Int
  priv bucket_rows : Int
} derive(Eq, Debug)

///|
pub extend Workload with Eq::{equal, not_equal}

///|
pub extend Workload with @debug.Debug::{to_repr}

///|
pub fn Workload::new(
  phase~ : String,
  query_tokens~ : Int,
  batch_rows~ : Int,
  prior_history~ : Int,
  bucket_tokens~ : Int,
  bucket_rows~ : Int,
) -> Workload? {
  guard (phase == "prefill" || phase == "decode" || phase == "mixed") &&
    batch_rows > 0 &&
    batch_rows <= query_tokens &&
    query_tokens <= bucket_tokens &&
    bucket_tokens <= 1048576 &&
    batch_rows <= bucket_rows &&
    bucket_rows <= bucket_tokens &&
    prior_history >= 0 &&
    prior_history <= 1048576 &&
    (phase != "decode" || query_tokens == batch_rows) else {
    return None
  }
  Some({
    phase,
    queries: query_tokens,
    rows: batch_rows,
    history: prior_history,
    bucket_queries: bucket_tokens,
    bucket_rows,
  })
}

///|
pub fn Workload::identity(self : Workload) -> String {
  "workload-v1:\{self.phase}:q\{self.queries}:r\{self.rows}:h\{self.history}:bq\{self.bucket_queries}:br\{self.bucket_rows}"
}

///|
pub fn Workload::fits(
  self : Workload,
  max_query_tokens~ : Int,
  max_batch_rows~ : Int,
) -> Bool {
  self.bucket_queries <= max_query_tokens && self.bucket_rows <= max_batch_rows
}

///|
/// Scope.device is a caller-declared performance compatibility class, not a
/// UUID. The class must include target, SM/cache/memory configuration and
/// relevant power/clock policy; ISA compatibility alone is insufficient.
/// Physical UUID remains provenance in the observation file, not this key.
pub fn Workload::scope(self : Workload, base : Scope) -> Scope? {
  Scope::new(
    device=base.device,
    toolchain=base.toolchain,
    shape=base.shape + ";" + self.identity(),
    program=base.program,
  )
}
