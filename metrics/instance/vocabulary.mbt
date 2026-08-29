///|
const INSTANCE_COUNTER_COUNT : Int = 22

///|
const INSTANCE_GAUGE_COUNT : Int = 6

///|
const INSTANCE_HISTOGRAM_COUNT : Int = 6

///|
const INSTANCE_HISTOGRAM_BUCKET_COUNT : Int = 16

///|
fn counter_index(counter : LunaInstanceCounter) -> Int {
  match counter {
    Admissions => 0
    Completions => 1
    Cancellations => 2
    Deadlines => 3
    Failures => 4
    PromptTokens => 5
    GeneratedTokens => 6
    WorkerRestarts => 7
    WorkerFailures => 8
    NetworkAccepts => 9
    NetworkDisconnects => 10
    NetworkRejections => 11
    Backpressure => 12
    PrefixLookups => 13
    PrefixHits => 14
    PrefixMisses => 15
    PrefixEvictions => 16
    PrefixTokensReused => 17
    PrefixTokensComputed => 18
    PrefixPublications => 19
    GraphHits => 20
    GraphMisses => 21
  }
}

///|
fn gauge_index(gauge : LunaInstanceGauge) -> Int {
  match gauge {
    QueueDepth => 0
    ActiveRequests => 1
    KvPagesUsed => 2
    KvPagesFree => 3
    PrefixEntries => 4
    PrefixPages => 5
  }
}

///|
fn histogram_index(histogram : LunaInstanceHistogram) -> Int {
  match histogram {
    RequestLatencyMillis => 0
    FirstTokenLatencyMillis => 1
    InterTokenLatencyMillis => 2
    ColdStartLatencyMillis => 3
    BatchRows => 4
    BatchTokens => 5
  }
}

///|
fn latency_bucket_bound(index : Int) -> Int {
  match index {
    0 => 1
    1 => 2
    2 => 5
    3 => 10
    4 => 20
    5 => 50
    6 => 100
    7 => 200
    8 => 500
    9 => 1000
    10 => 2000
    11 => 5000
    12 => 10000
    13 => 30000
    14 => 60000
    _ => 0x7fffffff
  }
}

///|
fn batch_rows_bucket_bound(index : Int) -> Int {
  match index {
    0 => 1
    1 => 2
    2 => 4
    3 => 8
    4 => 16
    5 => 32
    6 => 64
    7 => 128
    8 => 256
    9 => 512
    10 => 1024
    11 => 2048
    12 => 4096
    13 => 8192
    14 => 16384
    _ => 0x7fffffff
  }
}

///|
fn batch_tokens_bucket_bound(index : Int) -> Int {
  match index {
    0 => 1
    1 => 2
    2 => 4
    3 => 8
    4 => 16
    5 => 32
    6 => 64
    7 => 128
    8 => 256
    9 => 512
    10 => 1024
    11 => 2048
    12 => 4096
    13 => 8192
    14 => 32768
    _ => 0x7fffffff
  }
}

///|
pub fn luna_instance_counter_count() -> Int {
  INSTANCE_COUNTER_COUNT
}

///|
pub fn luna_instance_gauge_count() -> Int {
  INSTANCE_GAUGE_COUNT
}

///|
pub fn luna_instance_histogram_count() -> Int {
  INSTANCE_HISTOGRAM_COUNT
}

///|
pub fn luna_instance_histogram_bucket_count() -> Int {
  INSTANCE_HISTOGRAM_BUCKET_COUNT
}

///|
pub fn luna_instance_histogram_bucket_upper_bound(
  histogram : LunaInstanceHistogram,
  bucket_index : Int,
) -> Int raise LunaInstanceMetricsError {
  if bucket_index < 0 || bucket_index >= INSTANCE_HISTOGRAM_BUCKET_COUNT {
    raise Invalid(HistogramBucket)
  }
  histogram_bucket_upper_bound_unchecked(histogram, bucket_index)
}

///|
fn histogram_bucket_upper_bound_unchecked(
  histogram : LunaInstanceHistogram,
  bucket_index : Int,
) -> Int {
  match histogram {
    RequestLatencyMillis
    | FirstTokenLatencyMillis
    | InterTokenLatencyMillis
    | ColdStartLatencyMillis => latency_bucket_bound(bucket_index)
    BatchRows => batch_rows_bucket_bound(bucket_index)
    BatchTokens => batch_tokens_bucket_bound(bucket_index)
  }
}

///|
fn histogram_bucket_index(
  histogram : LunaInstanceHistogram,
  value : Int,
) -> Int {
  for index = 0; index < INSTANCE_HISTOGRAM_BUCKET_COUNT; index = index + 1 {
    if value <= histogram_bucket_upper_bound_unchecked(histogram, index) {
      break index
    }
  } nobreak {
    INSTANCE_HISTOGRAM_BUCKET_COUNT - 1
  }
}

///|
fn histogram_cell_index(
  histogram : LunaInstanceHistogram,
  bucket_index : Int,
) -> Int {
  histogram_index(histogram) * INSTANCE_HISTOGRAM_BUCKET_COUNT + bucket_index
}
