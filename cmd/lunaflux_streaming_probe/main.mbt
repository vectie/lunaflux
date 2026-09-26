///|
/// Diagnostic-only bounded polling. Production progress never busy-waits.
fn wait(
  cache : @host_cache.Cache,
  operation : @host_cache.Operation,
) -> @host_cache.Progress raise {
  let clock = @monotonic_clock.MonotonicClock::system()
  let start = clock.now_millis()
  for ;; {
    let result = cache.poll(operation)
    match result {
      Pending => ()
      _ => return result
    }
    if clock.now_millis() - start > 10000UL {
      fail("transfer timeout")
    }
  }
}

///|
fn layout() -> @device_layout.DeviceKvLayout raise {
  let digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  @device_layout.plan(
    @spec.ModelIdentity::new(
      @spec.ContentDigest::from_sha256(digest),
      @spec.PlanDigest::from_sha256(digest),
    ),
    @plan.KvCacheGeometry::new(
      layer_count=2,
      kv_head_count=2,
      head_dimension=4,
      dtype=BFloat16,
      layout=LayerMajorSplitKeyValueV1,
    ),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=8,
      total_page_count=4,
      max_arena_bytes=4096L,
    ),
  )
}

///|
fn exercise(
  allocation : @device.Allocation,
  cache : @host_cache.Cache,
  layout : @device_layout.DeviceKvLayout,
) -> Unit raise {
  let source = Bytes::makei(layout.arena_bytes().to_int(), i => {
    (i % 251).to_byte()
  })
  allocation.copy_from_host(source, 0L, 0L, layout.arena_bytes())
  guard cache.begin_spill(0) is Started(spill) else { fail("spill admission") }
  assert_true(wait(cache, spill) is HostReady)
  let clock = @monotonic_clock.MonotonicClock::system()
  let start = clock.now_millis()
  for _ in 0..<128 {
    guard cache.begin_restore(spill.entry(), 3) is Started(restore) else {
      fail("restore admission")
    }
    assert_true(wait(cache, restore) is DeviceReady)
    assert_true(cache.poll(restore) is Stale)
  }
  let elapsed = clock.now_millis() - start
  let restored = allocation.copy_to_host(0L, layout.arena_bytes())
  for layer in 0..<2 {
    for component in [@device_layout.Key, @device_layout.Value] {
      let src = layout.region(layer, component, 0)
      let dst = layout.region(layer, component, 3)
      for i in 0..<src.logical_bytes().to_int() {
        assert_eq(
          restored[dst.offset_bytes().to_int() + i],
          source[src.offset_bytes().to_int() + i],
        )
      }
      // Device alignment padding is not copied from another page.
      for i in dst.logical_bytes().to_int()..<dst.reserved_bytes().to_int() {
        assert_eq(
          restored[dst.offset_bytes().to_int() + i],
          source[dst.offset_bytes().to_int() + i],
        )
      }
    }
  }
  guard cache.begin_restore(spill.entry(), 2) is Started(cancelled) else {
    fail("cancel admission")
  }
  assert_true(cache.cancel(cancelled))
  assert_true(wait(cache, cancelled) is Cancelled)
  assert_true(cache.evict(spill.entry()))
  println("schema=lunaflux.streaming-page-probe.v1")
  println("physical_page_roundtrip_pass=true")
  println("restore_cycles=128")
  println("restore_bytes_per_cycle=\{cache.plan().slot_bytes()}")
  println("restore_total_milliseconds=\{elapsed}")
  println("model_logits_parity=not_run")
}

///|
fn main raise {
  let args = @env.args()
  guard args is [_, ordinal, "--separate-host-memory"] else {
    fail(
      "usage: lunaflux_streaming_probe ORDINAL --separate-host-memory; only for a discrete GPU",
    )
  }
  let geometry = layout()
  let plan = @host_cache.Plan::new(
    geometry,
    slots=2,
    transfer_lanes=2,
    host_budget=1024L,
    separate_host_memory=true,
  )
  let context = @device.open_context(@string.parse_int(ordinal))
  let allocation = {
    errdefer context.close()
    context.allocate(geometry.arena_bytes())
  }
  let cache = {
    errdefer {
      allocation.close()
      context.close()
    }
    @host_cache.Cache::prepare(context, allocation, plan)
  }
  let outcome : Result[Unit, Error] = try {
    guard cache.is_prepared() else { fail("streaming pool preparation failed") }
    exercise(allocation, cache, geometry)
    Ok(())
  } catch {
    error => Err(error)
  }
  // Explicit teardown is attempted while every owner remains in this scope.
  cache.close()
  allocation.close()
  context.close()
  match outcome {
    Ok(_) => println("resources_closed=true")
    Err(error) => raise error
  }
}
