///|
priv struct SpawnedDirectResources {
  queue_depth : Int
  active_requests : Int
  kv_pages_used : Int
  kv_pages_free : Int
  prefix_lookups : UInt64
  prefix_hits : UInt64
  prefix_misses : UInt64
  prefix_publications : UInt64
  prefix_tokens_reused : UInt64
  prefix_tokens_computed : UInt64
  prefix_entries : Int
  prefix_pages : Int
}

///|
fn capture_spawned_direct_resources(
  service : @online_session.LunaOnlineFramedService,
) -> SpawnedDirectResources raise {
  let telemetry = service.luna_telemetry()
  {
    queue_depth: telemetry.queue_depth(),
    active_requests: telemetry.active_requests(),
    kv_pages_used: telemetry.kv_pages_used(),
    kv_pages_free: telemetry.kv_pages_free(),
    prefix_lookups: telemetry.prefix_lookups(),
    prefix_hits: telemetry.prefix_hits(),
    prefix_misses: telemetry.prefix_misses(),
    prefix_publications: telemetry.prefix_publications(),
    prefix_tokens_reused: telemetry.prefix_tokens_reused(),
    prefix_tokens_computed: telemetry.prefix_tokens_computed(),
    prefix_entries: telemetry.prefix_entries(),
    prefix_pages: telemetry.prefix_pages(),
  }
}

///|
fn spawned_direct_resources_valid(
  sample : SpawnedDirectResources,
  initial_kv_pages_free : Int,
  expected_computed_tokens : UInt64,
) -> Bool {
  sample.queue_depth == 0 &&
  sample.active_requests == 0 &&
  sample.kv_pages_used == 0 &&
  sample.kv_pages_free == initial_kv_pages_free &&
  sample.prefix_lookups == 0UL &&
  sample.prefix_hits == 0UL &&
  sample.prefix_misses == 0UL &&
  sample.prefix_publications == 0UL &&
  sample.prefix_tokens_reused == 0UL &&
  sample.prefix_tokens_computed == expected_computed_tokens &&
  sample.prefix_entries == 0 &&
  sample.prefix_pages == 0
}

///|
/// Exact post-handoff direct-service ownership shape shared by physical
/// qualification modes that deliberately bind no listener.
fn spawned_direct_validation_owner_valid(owner : RuntimeInstanceOwner) -> Bool {
  owner.preparations.length() == 0 &&
  owner.cleanup.length() == 0 &&
  owner.services.length() == 1 &&
  owner.servers.length() == 0 &&
  owner.openai_servers.length() == 0 &&
  owner.borrowed_roots.length() == 0
}

///|
/// Retires one completed direct semantic stream without owning caller-specific
/// error vocabulary. The service must return to Ready before another request.
fn retire_spawned_direct_stream(
  service : @online_session.LunaOnlineFramedService,
  stream : @online_session.LunaOnlineFramedStream,
  turn_limit : Int,
) -> Bool {
  stream.disconnect() catch {
    _ => return false
  }
  for _ in 0..<turn_limit {
    if service.readiness() == LunaOnlineFramedServiceReady {
      return true
    }
    try stream.progress() catch {
      _ => return service.readiness() == LunaOnlineFramedServiceReady
    } noraise {
      _ => ()
    }
  }
  service.readiness() == LunaOnlineFramedServiceReady
}
