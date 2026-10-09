///|
fn streaming_cycle(
  pool : @residency.Pool,
  shapes : ArrayView[@streaming_policy.PrefillShape],
  cost : @streaming_policy.RestoreCost,
) -> Unit {
  guard pool.begin_spill() is Started(spill) else { alloc_probe_abort() }
  guard pool.complete(spill) is HostReady else { alloc_probe_abort() }
  guard pool.begin_restore(spill.entry()) is Started(restore) else {
    alloc_probe_abort()
  }
  guard pool.cancel(restore) is RetainUntilCompletion else {
    alloc_probe_abort()
  }
  guard pool.complete(restore) is RestoreCancelled else { alloc_probe_abort() }
  guard pool.begin_restore(spill.entry()) is Started(second) else {
    alloc_probe_abort()
  }
  guard pool.complete(restore) is Rejected else { alloc_probe_abort() }
  guard pool.complete(second) is DeviceReady else { alloc_probe_abort() }
  guard pool.evict(spill.entry()) is Released else { alloc_probe_abort() }
  guard pool.begin_spill() is Started(failed) else { alloc_probe_abort() }
  guard pool.fail(failed) is DrainRequired else { alloc_probe_abort() }
  guard pool.drained(failed) is Released else { alloc_probe_abort() }
  let selected = @streaming_policy.select_prefill(
    shapes,
    remaining_tokens=32,
    token_budget=32,
    workspace_budget=128L,
    available_pages=6,
    decode_reserve=2,
    tokens_per_page=8,
    tail_free_tokens=0,
  )
  if selected != 1 ||
    !cost.should_restore(
      queued_ns=10L,
      deadline_slack_ns=1000L,
      destination_reserved=true,
      transfer_credit=true,
    ) {
    alloc_probe_abort()
  }
}

///|
fn verify_streaming_allocations() -> Unit {
  let pool = @residency.Pool::new(
    try! @residency.Limits::new(
      slots=2,
      transfers=1,
      bytes_per_slot=64L,
      host_budget=128L,
      separate_host_memory=true,
    ),
  )
  let shapes = [
    try! @streaming_policy.PrefillShape::new(tokens=8, workspace_bytes=64L),
    try! @streaming_policy.PrefillShape::new(tokens=32, workspace_bytes=128L),
  ]
  let cost = try! @streaming_policy.RestoreCost::new(
    restore_ns=100L,
    recompute_ns=500L,
    minimum_saving_ns=50L,
  )
  streaming_cycle(pool, shapes, cost)
  alloc_probe_begin()
  for _ in 0..<10000 {
    streaming_cycle(pool, shapes, cost)
  }
  let observed = alloc_probe_end()
  if alloc_probe_check(observed, 1) != 1 || pool.available() != 2 {
    alloc_probe_abort()
  }
}
