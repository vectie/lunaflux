///|
fn saturating_add(current : UInt64, amount : Int) -> UInt64 {
  let increment = amount.to_uint64()
  if current > 0xffffffffffffffffUL - increment {
    0xffffffffffffffffUL
  } else {
    current + increment
  }
}

///|
fn saturating_add_u64(current : UInt64, amount : UInt64) -> UInt64 {
  if current > 0xffffffffffffffffUL - amount {
    0xffffffffffffffffUL
  } else {
    current + amount
  }
}

///|
pub fn LunaInstanceMetrics::new() -> LunaInstanceMetrics {
  {
    counters: FixedArray::make(INSTANCE_COUNTER_COUNT, 0UL),
    gauges: FixedArray::make(INSTANCE_GAUGE_COUNT, 0),
    histogram_buckets: FixedArray::make(
      INSTANCE_HISTOGRAM_COUNT * INSTANCE_HISTOGRAM_BUCKET_COUNT,
      0UL,
    ),
    snapshot_counters: FixedArray::make(INSTANCE_COUNTER_COUNT, 0UL),
    snapshot_gauges: FixedArray::make(INSTANCE_GAUGE_COUNT, 0),
    snapshot_histogram_buckets: FixedArray::make(
      INSTANCE_HISTOGRAM_COUNT * INSTANCE_HISTOGRAM_BUCKET_COUNT,
      0UL,
    ),
    snapshot_generation: 0UL,
  }
}

///|
pub fn LunaInstanceMetrics::record_counter(
  self : LunaInstanceMetrics,
  counter : LunaInstanceCounter,
  amount? : Int = 1,
) -> Unit raise LunaInstanceMetricsError {
  if amount < 0 {
    raise Invalid(CounterValue)
  }
  let index = counter_index(counter)
  self.counters[index] = saturating_add(self.counters[index], amount)
}

///|
/// Adds one already-nonnegative scalar counter delta and saturates at UInt64
/// maximum. This form is intended for monotonic lower-owner telemetry whose
/// native width must not be truncated through `Int`.
pub fn LunaInstanceMetrics::record_counter_u64(
  self : LunaInstanceMetrics,
  counter : LunaInstanceCounter,
  amount : UInt64,
) -> Unit {
  let index = counter_index(counter)
  self.counters[index] = saturating_add_u64(self.counters[index], amount)
}

///|
pub fn LunaInstanceMetrics::set_gauge(
  self : LunaInstanceMetrics,
  gauge : LunaInstanceGauge,
  value : Int,
) -> Unit raise LunaInstanceMetricsError {
  if value < 0 {
    raise Invalid(GaugeValue)
  }
  self.gauges[gauge_index(gauge)] = value
}

///|
pub fn LunaInstanceMetrics::record_histogram(
  self : LunaInstanceMetrics,
  histogram : LunaInstanceHistogram,
  value : Int,
) -> Unit raise LunaInstanceMetricsError {
  if value < 0 {
    raise Invalid(HistogramValue)
  }
  let index = histogram_cell_index(
    histogram,
    histogram_bucket_index(histogram, value),
  )
  self.histogram_buckets[index] = saturating_add(
    self.histogram_buckets[index],
    1,
  )
}

///|
pub fn LunaInstanceMetrics::counter_value(
  self : LunaInstanceMetrics,
  counter : LunaInstanceCounter,
) -> UInt64 {
  self.counters[counter_index(counter)]
}

///|
pub fn LunaInstanceMetrics::gauge_value(
  self : LunaInstanceMetrics,
  gauge : LunaInstanceGauge,
) -> Int {
  self.gauges[gauge_index(gauge)]
}

///|
pub fn LunaInstanceMetrics::histogram_bucket_value(
  self : LunaInstanceMetrics,
  histogram : LunaInstanceHistogram,
  bucket_index : Int,
) -> UInt64 raise LunaInstanceMetricsError {
  if bucket_index < 0 || bucket_index >= INSTANCE_HISTOGRAM_BUCKET_COUNT {
    raise Invalid(HistogramBucket)
  }
  self.histogram_buckets[histogram_cell_index(histogram, bucket_index)]
}

///|
pub fn LunaInstanceMetrics::reset(self : LunaInstanceMetrics) -> Unit {
  for index in 0..<INSTANCE_COUNTER_COUNT {
    self.counters[index] = 0UL
  }
  for index in 0..<INSTANCE_GAUGE_COUNT {
    self.gauges[index] = 0
  }
  for index in 0..<(INSTANCE_HISTOGRAM_COUNT * INSTANCE_HISTOGRAM_BUCKET_COUNT) {
    self.histogram_buckets[index] = 0UL
  }
}
