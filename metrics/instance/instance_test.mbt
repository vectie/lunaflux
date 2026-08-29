///|
test "public finite metric vocabulary records every scalar" {
  let metrics = @instance.LunaInstanceMetrics::new()
  let counters : Array[@instance.LunaInstanceCounter] = [
    Admissions,
    Completions,
    Cancellations,
    Deadlines,
    Failures,
    PromptTokens,
    GeneratedTokens,
    WorkerRestarts,
    WorkerFailures,
    NetworkAccepts,
    NetworkDisconnects,
    NetworkRejections,
    Backpressure,
    PrefixLookups,
    PrefixHits,
    PrefixMisses,
    PrefixEvictions,
    PrefixTokensReused,
    PrefixTokensComputed,
    PrefixPublications,
    GraphHits,
    GraphMisses,
  ]
  assert_eq(counters.length(), @instance.luna_instance_counter_count())
  for index, counter in counters {
    metrics.record_counter(counter, amount=index + 1)
    assert_eq(metrics.counter_value(counter), (index + 1).to_uint64())
  }
  let gauges : Array[@instance.LunaInstanceGauge] = [
    QueueDepth,
    ActiveRequests,
    KvPagesUsed,
    KvPagesFree,
    PrefixEntries,
    PrefixPages,
  ]
  assert_eq(gauges.length(), @instance.luna_instance_gauge_count())
  for index, gauge in gauges {
    metrics.set_gauge(gauge, index * 17)
    assert_eq(metrics.gauge_value(gauge), index * 17)
  }
}

///|
test "unsigned counter deltas preserve width and saturate" {
  let metrics = @instance.LunaInstanceMetrics::new()
  metrics.record_counter_u64(PrefixTokensComputed, 0xfffffffffffffffeUL)
  assert_eq(metrics.counter_value(PrefixTokensComputed), 0xfffffffffffffffeUL)
  metrics.record_counter_u64(PrefixTokensComputed, 7UL)
  assert_eq(metrics.counter_value(PrefixTokensComputed), 0xffffffffffffffffUL)
  metrics.record_counter_u64(PrefixTokensComputed, 0UL)
  assert_eq(metrics.counter_value(PrefixTokensComputed), 0xffffffffffffffffUL)
}

///|
test "every histogram exact boundary selects its fixed bucket" {
  let metrics = @instance.LunaInstanceMetrics::new()
  let histograms : Array[@instance.LunaInstanceHistogram] = [
    RequestLatencyMillis,
    FirstTokenLatencyMillis,
    InterTokenLatencyMillis,
    ColdStartLatencyMillis,
    BatchRows,
    BatchTokens,
  ]
  assert_eq(histograms.length(), @instance.luna_instance_histogram_count())
  assert_eq(@instance.luna_instance_histogram_bucket_count(), 16)
  for histogram in histograms {
    for bucket in 0..<@instance.luna_instance_histogram_bucket_count() {
      let bound = @instance.luna_instance_histogram_bucket_upper_bound(
        histogram, bucket,
      )
      metrics.record_histogram(histogram, bound)
      assert_eq(metrics.histogram_bucket_value(histogram, bucket), 1UL)
    }
    metrics.reset()
    metrics.record_histogram(histogram, 0)
    assert_eq(metrics.histogram_bucket_value(histogram, 0), 1UL)
    for bucket in 0..<15 {
      let bound = @instance.luna_instance_histogram_bucket_upper_bound(
        histogram, bucket,
      )
      metrics.record_histogram(histogram, bound + 1)
      assert_eq(metrics.histogram_bucket_value(histogram, bucket + 1), 1UL)
    }
    metrics.reset()
  }
}

///|
test "negative values and bucket indices are typed and pre-mutation" {
  let metrics = @instance.LunaInstanceMetrics::new()
  try metrics.record_counter(Admissions, amount=-1) catch {
    Invalid(CounterValue) => ()
    _ => fail("unexpected negative counter error")
  } noraise {
    _ => fail("negative counter amount was accepted")
  }
  try metrics.set_gauge(QueueDepth, -1) catch {
    Invalid(GaugeValue) => ()
    _ => fail("unexpected negative gauge error")
  } noraise {
    _ => fail("negative gauge was accepted")
  }
  try metrics.record_histogram(RequestLatencyMillis, -1) catch {
    Invalid(HistogramValue) => ()
    _ => fail("unexpected negative histogram error")
  } noraise {
    _ => fail("negative histogram value was accepted")
  }
  try metrics.histogram_bucket_value(RequestLatencyMillis, -1) catch {
    Invalid(HistogramBucket) => ()
    _ => fail("unexpected negative bucket error")
  } noraise {
    _ => fail("negative bucket was accepted")
  }
  try
    @instance.luna_instance_histogram_bucket_upper_bound(
      RequestLatencyMillis,
      16,
    )
  catch {
    Invalid(HistogramBucket) => ()
    _ => fail("unexpected oversized bucket error")
  } noraise {
    _ => fail("oversized bucket was accepted")
  }
  assert_eq(metrics.counter_value(Admissions), 0UL)
  assert_eq(metrics.gauge_value(QueueDepth), 0)
  assert_eq(metrics.histogram_bucket_value(RequestLatencyMillis, 0), 0UL)
}

///|
test "snapshot is stable through mutation and stale after slot reuse" {
  let metrics = @instance.LunaInstanceMetrics::new()
  metrics.record_counter(Admissions, amount=7)
  metrics.set_gauge(ActiveRequests, 3)
  metrics.record_histogram(FirstTokenLatencyMillis, 20)
  let first = metrics.snapshot()
  assert_true(first.is_live())
  metrics.record_counter(Admissions, amount=5)
  metrics.set_gauge(ActiveRequests, 9)
  metrics.reset()
  assert_eq(first.counter_value(Admissions), 7UL)
  assert_eq(first.gauge_value(ActiveRequests), 3)
  assert_eq(first.histogram_bucket_value(FirstTokenLatencyMillis, 4), 1UL)
  let second = metrics.snapshot()
  assert_false(first.is_live())
  assert_true(second.is_live())
  try first.counter_value(Admissions) catch {
    StaleSnapshot => ()
    _ => fail("unexpected stale snapshot error")
  } noraise {
    _ => fail("reused snapshot slot kept its old authority")
  }
  try first.histogram_bucket_value(FirstTokenLatencyMillis, -1) catch {
    StaleSnapshot => ()
    _ => fail("stale authentication lost precedence over bucket bounds")
  } noraise {
    _ => fail("stale snapshot accepted an invalid bucket")
  }
  assert_eq(second.counter_value(Admissions), 0UL)
  assert_eq(second.gauge_value(ActiveRequests), 0)
}
