///|
test "framed telemetry carries every scheduler prefix scalar" {
  let telemetry : LunaOnlineFramedTelemetry = {
    queue_depth: 7,
    active_requests: 3,
    kv_pages_used: 11,
    kv_pages_free: 13,
    prefix_lookups: 17UL,
    prefix_hits: 19UL,
    prefix_misses: 23UL,
    prefix_evictions: 29UL,
    prefix_tokens_reused: 31UL,
    prefix_tokens_computed: 37UL,
    prefix_publications: 41UL,
    prefix_entries: 43,
    prefix_pages: 47,
  }
  assert_eq(telemetry.queue_depth(), 7)
  assert_eq(telemetry.active_requests(), 3)
  assert_eq(telemetry.kv_pages_used(), 11)
  assert_eq(telemetry.kv_pages_free(), 13)
  assert_eq(telemetry.prefix_lookups(), 17UL)
  assert_eq(telemetry.prefix_hits(), 19UL)
  assert_eq(telemetry.prefix_misses(), 23UL)
  assert_eq(telemetry.prefix_evictions(), 29UL)
  assert_eq(telemetry.prefix_tokens_reused(), 31UL)
  assert_eq(telemetry.prefix_tokens_computed(), 37UL)
  assert_eq(telemetry.prefix_publications(), 41UL)
  assert_eq(telemetry.prefix_entries(), 43)
  assert_eq(telemetry.prefix_pages(), 47)
}
