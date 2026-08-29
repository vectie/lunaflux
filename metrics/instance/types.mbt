///|
#valtype
pub(all) enum LunaInstanceCounter {
  Admissions
  Completions
  Cancellations
  Deadlines
  Failures
  PromptTokens
  GeneratedTokens
  WorkerRestarts
  WorkerFailures
  NetworkAccepts
  NetworkDisconnects
  NetworkRejections
  Backpressure
  PrefixLookups
  PrefixHits
  PrefixMisses
  PrefixEvictions
  PrefixTokensReused
  PrefixTokensComputed
  PrefixPublications
  GraphHits
  GraphMisses
}

///|
#valtype
pub(all) enum LunaInstanceGauge {
  QueueDepth
  ActiveRequests
  KvPagesUsed
  KvPagesFree
  PrefixEntries
  PrefixPages
}

///|
#valtype
pub(all) enum LunaInstanceHistogram {
  RequestLatencyMillis
  FirstTokenLatencyMillis
  InterTokenLatencyMillis
  ColdStartLatencyMillis
  BatchRows
  BatchTokens
}

///|
pub(all) enum LunaInstanceMetricRule {
  CounterValue
  GaugeValue
  HistogramValue
  HistogramBucket
  Snapshot
}

///|
pub(all) suberror LunaInstanceMetricsError {
  Invalid(LunaInstanceMetricRule)
  StaleSnapshot
  SnapshotGenerationExhausted
}

///|
/// Startup-owned fixed-capacity telemetry storage for one engine instance.
pub struct LunaInstanceMetrics {
  priv counters : FixedArray[UInt64]
  priv gauges : FixedArray[Int]
  priv histogram_buckets : FixedArray[UInt64]
  priv snapshot_counters : FixedArray[UInt64]
  priv snapshot_gauges : FixedArray[Int]
  priv snapshot_histogram_buckets : FixedArray[UInt64]
  priv mut snapshot_generation : UInt64
}

///|
/// Opaque generation-authenticated view of the owner's one snapshot slot.
#valtype
pub struct LunaInstanceMetricsSnapshot {
  priv owner : LunaInstanceMetrics
  priv generation : UInt64
}
