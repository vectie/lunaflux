///|
const LUNA_ONLINE_CONTROL_METRICS_RESPONSE_CAPACITY : Int = 4096

///|
const LUNA_ONLINE_CONTROL_METRICS_BODY_OFFSET : Int = 256

///|
fn luna_online_metrics_write_bytes(
  write : LunaOnlineTcpOutputWrite,
  cursor : Int,
  value : Bytes,
) -> Int raise {
  let scratch = write.require_mutating()
  if cursor < 0 || cursor > scratch.capacity - value.length() {
    raise LunaOnlineTcpScratchRange
  }
  for index in 0..<value.length() {
    scratch.mutable[cursor + index] = value[index]
  }
  cursor + value.length()
}

///|
fn luna_online_metrics_write_u64(
  write : LunaOnlineTcpOutputWrite,
  cursor : Int,
  value : UInt64,
) -> Int raise {
  let scratch = write.require_mutating()
  let mut digits = 1
  let mut remaining = value
  while remaining >= 10UL {
    remaining = remaining / 10UL
    digits += 1
  }
  if cursor < 0 || cursor > scratch.capacity - digits {
    raise LunaOnlineTcpScratchRange
  }
  let end = cursor + digits
  remaining = value
  for index = end - 1; index >= cursor; index = index - 1 {
    scratch.mutable[index] = (remaining % 10UL).to_byte() + b'0'
    remaining = remaining / 10UL
  }
  end
}

///|
fn luna_online_metrics_write_int(
  write : LunaOnlineTcpOutputWrite,
  cursor : Int,
  value : Int,
) -> Int raise {
  if value < 0 {
    raise LunaOnlineTcpScratchRange
  }
  luna_online_metrics_write_u64(write, cursor, value.to_uint64())
}

///|
fn luna_online_metrics_write_counter(
  write : LunaOnlineTcpOutputWrite,
  snapshot : @metrics.LunaInstanceMetricsSnapshot,
  cursor : Int,
  first~ : Bool,
  name : Bytes,
  counter : @metrics.LunaInstanceCounter,
) -> Int raise {
  let cursor = luna_online_metrics_write_bytes(
    write,
    cursor,
    if first {
      b"\""
    } else {
      b",\""
    },
  )
  let cursor = luna_online_metrics_write_bytes(write, cursor, name)
  let cursor = luna_online_metrics_write_bytes(write, cursor, b"\":")
  luna_online_metrics_write_u64(write, cursor, snapshot.counter_value(counter))
}

///|
fn luna_online_metrics_write_gauge(
  write : LunaOnlineTcpOutputWrite,
  snapshot : @metrics.LunaInstanceMetricsSnapshot,
  cursor : Int,
  first~ : Bool,
  name : Bytes,
  gauge : @metrics.LunaInstanceGauge,
) -> Int raise {
  let cursor = luna_online_metrics_write_bytes(
    write,
    cursor,
    if first {
      b"\""
    } else {
      b",\""
    },
  )
  let cursor = luna_online_metrics_write_bytes(write, cursor, name)
  let cursor = luna_online_metrics_write_bytes(write, cursor, b"\":")
  luna_online_metrics_write_int(write, cursor, snapshot.gauge_value(gauge))
}

///|
fn luna_online_metrics_write_counters(
  write : LunaOnlineTcpOutputWrite,
  snapshot : @metrics.LunaInstanceMetricsSnapshot,
  cursor : Int,
) -> Int raise {
  let cursor = luna_online_metrics_write_counter(
    write,
    snapshot,
    cursor,
    first=true,
    b"admissions",
    Admissions,
  )
  let cursor = luna_online_metrics_write_counter(
    write,
    snapshot,
    cursor,
    first=false,
    b"completions",
    Completions,
  )
  let cursor = luna_online_metrics_write_counter(
    write,
    snapshot,
    cursor,
    first=false,
    b"cancellations",
    Cancellations,
  )
  let cursor = luna_online_metrics_write_counter(
    write,
    snapshot,
    cursor,
    first=false,
    b"deadlines",
    Deadlines,
  )
  let cursor = luna_online_metrics_write_counter(
    write,
    snapshot,
    cursor,
    first=false,
    b"failures",
    Failures,
  )
  let cursor = luna_online_metrics_write_counter(
    write,
    snapshot,
    cursor,
    first=false,
    b"prompt_tokens",
    PromptTokens,
  )
  let cursor = luna_online_metrics_write_counter(
    write,
    snapshot,
    cursor,
    first=false,
    b"generated_tokens",
    GeneratedTokens,
  )
  let cursor = luna_online_metrics_write_counter(
    write,
    snapshot,
    cursor,
    first=false,
    b"worker_restarts",
    WorkerRestarts,
  )
  let cursor = luna_online_metrics_write_counter(
    write,
    snapshot,
    cursor,
    first=false,
    b"worker_failures",
    WorkerFailures,
  )
  let cursor = luna_online_metrics_write_counter(
    write,
    snapshot,
    cursor,
    first=false,
    b"network_accepts",
    NetworkAccepts,
  )
  let cursor = luna_online_metrics_write_counter(
    write,
    snapshot,
    cursor,
    first=false,
    b"network_disconnects",
    NetworkDisconnects,
  )
  let cursor = luna_online_metrics_write_counter(
    write,
    snapshot,
    cursor,
    first=false,
    b"network_rejections",
    NetworkRejections,
  )
  let cursor = luna_online_metrics_write_counter(
    write,
    snapshot,
    cursor,
    first=false,
    b"backpressure",
    Backpressure,
  )
  let cursor = luna_online_metrics_write_counter(
    write,
    snapshot,
    cursor,
    first=false,
    b"prefix_lookups",
    PrefixLookups,
  )
  let cursor = luna_online_metrics_write_counter(
    write,
    snapshot,
    cursor,
    first=false,
    b"prefix_hits",
    PrefixHits,
  )
  let cursor = luna_online_metrics_write_counter(
    write,
    snapshot,
    cursor,
    first=false,
    b"prefix_misses",
    PrefixMisses,
  )
  let cursor = luna_online_metrics_write_counter(
    write,
    snapshot,
    cursor,
    first=false,
    b"prefix_evictions",
    PrefixEvictions,
  )
  let cursor = luna_online_metrics_write_counter(
    write,
    snapshot,
    cursor,
    first=false,
    b"prefix_tokens_reused",
    PrefixTokensReused,
  )
  let cursor = luna_online_metrics_write_counter(
    write,
    snapshot,
    cursor,
    first=false,
    b"prefix_tokens_computed",
    PrefixTokensComputed,
  )
  let cursor = luna_online_metrics_write_counter(
    write,
    snapshot,
    cursor,
    first=false,
    b"prefix_publications",
    PrefixPublications,
  )
  let cursor = luna_online_metrics_write_counter(
    write,
    snapshot,
    cursor,
    first=false,
    b"graph_hits",
    GraphHits,
  )
  luna_online_metrics_write_counter(
    write,
    snapshot,
    cursor,
    first=false,
    b"graph_misses",
    GraphMisses,
  )
}

///|
fn luna_online_metrics_write_gauges(
  write : LunaOnlineTcpOutputWrite,
  snapshot : @metrics.LunaInstanceMetricsSnapshot,
  cursor : Int,
) -> Int raise {
  let cursor = luna_online_metrics_write_gauge(
    write,
    snapshot,
    cursor,
    first=true,
    b"queue_depth",
    QueueDepth,
  )
  let cursor = luna_online_metrics_write_gauge(
    write,
    snapshot,
    cursor,
    first=false,
    b"active_requests",
    ActiveRequests,
  )
  let cursor = luna_online_metrics_write_gauge(
    write,
    snapshot,
    cursor,
    first=false,
    b"kv_pages_used",
    KvPagesUsed,
  )
  let cursor = luna_online_metrics_write_gauge(
    write,
    snapshot,
    cursor,
    first=false,
    b"kv_pages_free",
    KvPagesFree,
  )
  let cursor = luna_online_metrics_write_gauge(
    write,
    snapshot,
    cursor,
    first=false,
    b"prefix_entries",
    PrefixEntries,
  )
  luna_online_metrics_write_gauge(
    write,
    snapshot,
    cursor,
    first=false,
    b"prefix_pages",
    PrefixPages,
  )
}

///|
fn luna_online_render_control_metrics_response(
  write : LunaOnlineTcpOutputWrite,
  snapshot : @metrics.LunaInstanceMetricsSnapshot,
  cold_start_latency_millis : Int,
  readiness : Bool,
) -> Int raise {
  let published = cold_start_latency_millis >= 0 &&
    cold_start_latency_millis <= 86400000
  let ready = published && readiness
  let cursor = luna_online_metrics_write_bytes(
    write,
    LUNA_ONLINE_CONTROL_METRICS_BODY_OFFSET,
    b"{\"schema\":\"lunaflux.runtime.metrics.v1\",\"cold_start_latency_millis\":",
  )
  let cursor = if published {
    luna_online_metrics_write_int(write, cursor, cold_start_latency_millis)
  } else {
    luna_online_metrics_write_bytes(write, cursor, b"null")
  }
  let cursor = luna_online_metrics_write_bytes(
    write, cursor, b",\"readiness\":",
  )
  let cursor = luna_online_metrics_write_bytes(
    write,
    cursor,
    if ready {
      b"true"
    } else {
      b"false"
    },
  )
  let cursor = luna_online_metrics_write_bytes(
    write, cursor, b",\"counters\":{",
  )
  let cursor = luna_online_metrics_write_counters(write, snapshot, cursor)
  let cursor = luna_online_metrics_write_bytes(write, cursor, b"},\"gauges\":{")
  let cursor = luna_online_metrics_write_gauges(write, snapshot, cursor)
  let cursor = luna_online_metrics_write_bytes(write, cursor, b"}}\n")
  let body_length = cursor - LUNA_ONLINE_CONTROL_METRICS_BODY_OFFSET
  let header_cursor = luna_online_metrics_write_bytes(
    write,
    0,
    if ready {
      b"HTTP/1.1 200 OK\r\n"
    } else {
      b"HTTP/1.1 503 Service Unavailable\r\n"
    },
  )
  let header_cursor = luna_online_metrics_write_bytes(
    write, header_cursor, b"Content-Type: application/json\r\nCache-Control: no-store\r\nConnection: close\r\nContent-Length: ",
  )
  let header_cursor = luna_online_metrics_write_int(
    write, header_cursor, body_length,
  )
  let header_length = luna_online_metrics_write_bytes(
    write, header_cursor, b"\r\n\r\n",
  )
  let scratch = write.require_mutating()
  if header_length > LUNA_ONLINE_CONTROL_METRICS_BODY_OFFSET ||
    header_length > scratch.capacity - body_length {
    raise LunaOnlineTcpScratchRange
  }
  for index in 0..<body_length {
    scratch.mutable[header_length + index] = scratch.mutable[LUNA_ONLINE_CONTROL_METRICS_BODY_OFFSET +
      index]
  }
  header_length + body_length
}
