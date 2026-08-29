///|
pub fn LunaInstanceMetrics::snapshot(
  self : LunaInstanceMetrics,
) -> LunaInstanceMetricsSnapshot raise LunaInstanceMetricsError {
  if self.snapshot_generation == 0xffffffffffffffffUL {
    raise SnapshotGenerationExhausted
  }
  for index in 0..<INSTANCE_COUNTER_COUNT {
    self.snapshot_counters[index] = self.counters[index]
  }
  for index in 0..<INSTANCE_GAUGE_COUNT {
    self.snapshot_gauges[index] = self.gauges[index]
  }
  for index in 0..<(INSTANCE_HISTOGRAM_COUNT * INSTANCE_HISTOGRAM_BUCKET_COUNT) {
    self.snapshot_histogram_buckets[index] = self.histogram_buckets[index]
  }
  let generation = self.snapshot_generation + 1UL
  self.snapshot_generation = generation
  { owner: self, generation }
}

///|
pub fn LunaInstanceMetricsSnapshot::is_live(
  self : LunaInstanceMetricsSnapshot,
) -> Bool {
  self.generation != 0UL && self.owner.snapshot_generation == self.generation
}

///|
fn LunaInstanceMetricsSnapshot::require_live(
  self : LunaInstanceMetricsSnapshot,
) -> Unit raise LunaInstanceMetricsError {
  if !self.is_live() {
    raise StaleSnapshot
  }
}

///|
pub fn LunaInstanceMetricsSnapshot::counter_value(
  self : LunaInstanceMetricsSnapshot,
  counter : LunaInstanceCounter,
) -> UInt64 raise LunaInstanceMetricsError {
  self.require_live()
  self.owner.snapshot_counters[counter_index(counter)]
}

///|
pub fn LunaInstanceMetricsSnapshot::gauge_value(
  self : LunaInstanceMetricsSnapshot,
  gauge : LunaInstanceGauge,
) -> Int raise LunaInstanceMetricsError {
  self.require_live()
  self.owner.snapshot_gauges[gauge_index(gauge)]
}

///|
pub fn LunaInstanceMetricsSnapshot::histogram_bucket_value(
  self : LunaInstanceMetricsSnapshot,
  histogram : LunaInstanceHistogram,
  bucket_index : Int,
) -> UInt64 raise LunaInstanceMetricsError {
  self.require_live()
  if bucket_index < 0 || bucket_index >= INSTANCE_HISTOGRAM_BUCKET_COUNT {
    raise Invalid(HistogramBucket)
  }
  self.owner.snapshot_histogram_buckets[histogram_cell_index(
    histogram, bucket_index,
  )]
}
