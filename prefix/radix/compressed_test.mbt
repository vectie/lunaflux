///|
test "compressed branches share labels and partial lookup uses descendant anchor" {
  let index = try! @radix.PrefixIndex::new(index_limits(nodes=16, pages=8))
  let cache_identity = identity("a", "b", "c", b"scope")
  let pages = physical_pages(4)
  let first = publish_prefix(index, cache_identity, [1, 2, 3, 4], pages[0:2])
  ignore(publish_prefix(index, cache_identity, [1, 2, 5, 6], pages[2:4]))
  assert_eq(index.node_count(), 3)
  assert_eq(index.token_cell_count(), 6)
  let result = @radix.PrefixLookupBuffer::new(2)
  lookup_prefix(index, cache_identity, [1, 2, 9], result)
  assert_eq(result.matched_tokens(), 2)
  assert_eq(result.page_at(0), pages[0])
  assert_eq(result.entry(), Some(first))
  index.debug_check_invariants()
}

///|
test "priority precedes LRU and active descendant preserves shared anchors" {
  let index = try! @radix.PrefixIndex::new(
    index_limits(entries=4, nodes=16, pages=8),
  )
  let cache_identity = identity("a", "b", "c", b"scope")
  let pages = physical_pages(5)
  let long = publish_prefix(index, cache_identity, [1, 2, 3, 4], pages[0:2])
  let short = publish_prefix(index, cache_identity, [7, 8], pages[2:3])
  let eviction = @radix.PrefixEvictionBuffer::new(4)
  assert_eq(evict_prefix(index, eviction), short)
  assert_eq(eviction.page_count(), 1)
  index.retain_active(long)
  let ancestor = publish_prefix(index, cache_identity, [1, 2], pages[3:4])
  assert_eq(evict_prefix(index, eviction), ancestor)
  assert_eq(eviction.page_count(), 0)
  let lookup = @radix.PrefixLookupBuffer::new(2)
  lookup_prefix(index, cache_identity, [1, 2, 3, 4], lookup)
  assert_eq(lookup.page_at(0), pages[0])
  index.release_active(long)
  index.debug_check_invariants()
}

///|
test "equal-priority eviction uses deterministic LRU recency" {
  let index = @radix.PrefixIndex::new(index_limits(entries=3, nodes=8, pages=3))
  let cache_identity = identity("a", "b", "c", b"scope")
  let pages = physical_pages(2)
  let first = publish_prefix(index, cache_identity, [1, 2], pages[0:1])
  let second = publish_prefix(index, cache_identity, [3, 4], pages[1:2])
  index.touch(first)
  let eviction = @radix.PrefixEvictionBuffer::new(1)
  assert_eq(evict_prefix(index, eviction), second)
  assert_eq(eviction.page_at(0), pages[1])
  index.debug_check_invariants()
}

///|
fn planned_request_key(
  identity : TestRootIdentity,
  permission? : @inference.CachePermission = ReadWrite,
) -> (@radix.PrefixRequestKey, @inference.LunaRequestSemanticLease) raise {
  let limits = inference_limits()
  let storage = @inference.LunaRequestSemanticStorage::new(
    limits,
    @inference.LunaRequestSemanticStepBudget::new(17),
  )
  let write = storage.begin(
    stop_token_count=0,
    stop_string_count=0,
    cache_scope_byte_count=5,
    cache_permission=permission,
  )
  for byte in b"scope" {
    write.push_cache_scope_byte(byte)
  }
  let work = write.finish()
  while work.progress().is_pending() {

  }
  let lease = work.take_lease()
  (
    @radix.PrefixRequestKey::new(
      model_identity=identity.model,
      tokenizer=identity.tokenizer,
      layout=identity.layout,
      semantic=lease.prefix_view(),
    ),
    lease,
  )
}

///|
fn planned_publish(
  index : @radix.PrefixIndex,
  key : @radix.PrefixRequestKey,
  tokens : @radix.PrefixTokenBuffer,
  pages : @radix.PrefixPublishBuffer,
  priority~ : Int,
) -> @radix.PrefixPublishPlan raise @radix.PrefixIndexError {
  let start = index.try_plan_publish(key, tokens, pages, priority~)
  if !start.is_planned() {
    abort("expected publication plan")
  }
  start.plan()
}

///|
fn planned_eviction(
  index : @radix.PrefixIndex,
  output : @radix.PrefixEvictionBuffer,
) -> @radix.PrefixEvictionPlan raise @radix.PrefixIndexError {
  let start = index.try_plan_eviction(output)
  if !start.is_planned() {
    abort("expected eviction plan")
  }
  start.plan()
}

///|
test "production radix independently enforces cache permission" {
  let index = @radix.PrefixIndex::new(index_limits())
  let cache_identity = identity("a", "b", "c", b"scope")
  let tokens = @radix.PrefixTokenBuffer::new(2)
  tokens.push_token(1)
  tokens.push_token(2)
  let pages = @radix.PrefixPublishBuffer::new(1)
  pages.push_page(physical_pages(1)[0])
  let (write_key, write_lease) = planned_request_key(cache_identity)
  let plan = planned_publish(index, write_key, tokens, pages, priority=1)
  ignore(index.publish(plan, tokens, pages, priority=1))
  write_lease.release()

  let output = @radix.PrefixLookupBuffer::new(1)
  let (read_key, read_lease) = planned_request_key(
    cache_identity,
    permission=ReadOnly,
  )
  index.lookup(read_key, tokens, maximum_reusable_tokens=2, output)
  assert_true(output.has_entry())
  try index.try_plan_publish(read_key, tokens, pages, priority=1) catch {
    CachePermissionDenied => ()
    _ => fail("unexpected read-only publication error")
  } noraise {
    _ => fail("read-only key opened publication authority")
  }
  read_lease.release()

  let (disabled_key, disabled_lease) = planned_request_key(
    cache_identity,
    permission=Disabled,
  )
  index.lookup(disabled_key, tokens, maximum_reusable_tokens=2, output)
  assert_false(output.has_entry())
  try index.try_plan_publish(disabled_key, tokens, pages, priority=1) catch {
    CachePermissionDenied => ()
    _ => fail("unexpected disabled publication error")
  } noraise {
    _ => fail("disabled key opened publication authority")
  }
  disabled_lease.release()
  index.debug_check_invariants()
}

///|
test "publication plan authenticates exact caller buffers and is one shot" {
  let index = @radix.PrefixIndex::new(index_limits())
  let cache_identity = identity("a", "b", "c", b"scope")
  let (key, lease) = planned_request_key(cache_identity)
  let physical = physical_pages(2)
  let tokens = @radix.PrefixTokenBuffer::new(2)
  tokens.push_token(1)
  tokens.push_token(2)
  let pages = @radix.PrefixPublishBuffer::new(1)
  pages.push_page(physical[0])
  let token_plan = planned_publish(index, key, tokens, pages, priority=1)
  try index.try_plan_publish(key, tokens, pages, priority=1) catch {
    PublishPlanBusy => ()
    _ => fail("unexpected overlapping publication-plan error")
  } noraise {
    _ => fail("overlapping publication plan was accepted")
  }
  try index.try_plan_publish(key, tokens, pages, priority=1) catch {
    PublishPlanBusy => ()
    _ => fail("unexpected publication overlap error")
  } noraise {
    _ => fail("publication preflight bypassed a live plan")
  }
  tokens.clear()
  tokens.push_token(3)
  tokens.push_token(4)
  try index.publish(token_plan, tokens, pages, priority=1) catch {
    PublishPlanInvalidated => ()
    _ => fail("unexpected changed-token publication error")
  } noraise {
    _ => fail("changed publication tokens were accepted")
  }
  index.abort_publish(token_plan)

  tokens.clear()
  tokens.push_token(1)
  tokens.push_token(2)
  let page_plan = planned_publish(index, key, tokens, pages, priority=1)
  try index.abort_publish(token_plan) catch {
    StalePublishPlan => ()
    _ => fail("unexpected reused publication-plan alias error")
  } noraise {
    _ => fail("stale plan alias aborted a newer transaction")
  }
  pages.clear()
  pages.push_page(physical[1])
  try index.publish(page_plan, tokens, pages, priority=1) catch {
    PublishPlanInvalidated => ()
    _ => fail("unexpected changed-page publication error")
  } noraise {
    _ => fail("changed publication pages were accepted")
  }
  index.abort_publish(page_plan)

  pages.clear()
  pages.push_page(physical[0])
  let accepted = planned_publish(index, key, tokens, pages, priority=1)
  ignore(index.publish(accepted, tokens, pages, priority=1))
  try index.abort_publish(accepted) catch {
    StalePublishPlan => ()
    _ => fail("unexpected consumed publication-plan error")
  } noraise {
    _ => fail("consumed publication plan remained live")
  }
  lease.release()
  index.debug_check_invariants()
}

///|
test "stale planned identity fails before mutation and retains abort authority" {
  let index = @radix.PrefixIndex::new(index_limits())
  let cache_identity = identity("a", "b", "c", b"scope")
  let (key, lease) = planned_request_key(cache_identity)
  let tokens = @radix.PrefixTokenBuffer::new(2)
  tokens.push_token(1)
  tokens.push_token(2)
  let pages = @radix.PrefixPublishBuffer::new(1)
  pages.push_page(physical_pages(1)[0])
  let plan = planned_publish(index, key, tokens, pages, priority=1)
  lease.release()
  try index.publish(plan, tokens, pages, priority=1) catch {
    StaleKey => ()
    _ => fail("unexpected stale planned-identity error")
  } noraise {
    _ => fail("stale planned identity reached radix mutation")
  }
  assert_eq(index.entry_count(), 0)
  index.abort_publish(plan)
  index.debug_check_invariants()
}

///|
test "eviction plan rejects substituted output overlap and stale aliases" {
  let index = @radix.PrefixIndex::new(index_limits(entries=3, pages=3))
  let cache_identity = identity("a", "b", "c", b"scope")
  let physical = physical_pages(2)
  let first = publish_prefix(index, cache_identity, [1, 2], physical[0:1])
  ignore(publish_prefix(index, cache_identity, [3, 4], physical[1:2]))
  let output = @radix.PrefixEvictionBuffer::new(1)
  let plan = planned_eviction(index, output)
  let (key, lease) = planned_request_key(cache_identity)
  let publish_tokens = @radix.PrefixTokenBuffer::new(2)
  publish_tokens.push_token(5)
  publish_tokens.push_token(6)
  let publish_pages = @radix.PrefixPublishBuffer::new(1)
  publish_pages.push_page(physical[0])
  try
    index.try_plan_publish(key, publish_tokens, publish_pages, priority=1)
  catch {
    PublishPlanBusy => ()
    _ => fail("unexpected publication overlap error")
  } noraise {
    _ => fail("publication overlapped an exact eviction plan")
  }
  output.clear()
  try index.commit_eviction(plan, output) catch {
    EvictionPlanInvalidated => ()
    _ => fail("unexpected substituted eviction-output error")
  } noraise {
    _ => fail("substituted eviction output reached metadata mutation")
  }
  assert_eq(index.entry_count(), 2)
  index.abort_eviction(plan)

  let stale = planned_eviction(index, output)
  index.abort_eviction(stale)
  let accepted = planned_eviction(index, output)
  try index.abort_eviction(stale) catch {
    StaleEvictionPlan => ()
    _ => fail("unexpected stale eviction-plan alias error")
  } noraise {
    _ => fail("stale eviction plan aborted a newer transaction")
  }
  assert_eq(index.commit_eviction(accepted, output), first)
  lease.release()
  index.debug_check_invariants()
}

///|
test "warmed publication and eviction pressure use scalar start statuses" {
  let index = @radix.PrefixIndex::new(index_limits(entries=1, pages=1))
  let cache_identity = identity("a", "b", "c", b"scope")
  let (key, lease) = planned_request_key(cache_identity)
  let tokens = @radix.PrefixTokenBuffer::new(2)
  tokens.push_token(1)
  tokens.push_token(2)
  let pages = @radix.PrefixPublishBuffer::new(1)
  pages.push_page(physical_pages(1)[0])
  let first = index.try_plan_publish(key, tokens, pages, priority=1)
  assert_true(first.is_planned())
  ignore(index.publish(first.plan(), tokens, pages, priority=1))

  let duplicate = index.try_plan_publish(key, tokens, pages, priority=1)
  assert_true(duplicate.is_duplicate())
  assert_eq(duplicate.duplicate_entry().index(), 0)
  tokens.clear()
  tokens.push_token(3)
  tokens.push_token(4)
  let full = index.try_plan_publish(key, tokens, pages, priority=1)
  assert_true(full.is_capacity_exhausted())

  let eviction = @radix.PrefixEvictionBuffer::new(1)
  let selected = index.try_plan_eviction(eviction)
  assert_true(selected.is_planned())
  ignore(index.commit_eviction(selected.plan(), eviction))
  let empty = index.try_plan_eviction(eviction)
  assert_true(empty.is_absent())

  let narrow = @radix.PrefixIndex::new(
    index_limits(entries=1, pages=1, max_scope=4),
  )
  assert_true(
    narrow.try_plan_publish(key, tokens, pages, priority=1).is_rejected(),
  )
  lease.release()
  index.debug_check_invariants()
}

///|
test "identical compressed churn replays the same scalar snapshot" {
  let left = @radix.PrefixIndex::new(index_limits(entries=4, nodes=16, pages=8))
  let right = @radix.PrefixIndex::new(
    index_limits(entries=4, nodes=16, pages=8),
  )
  let cache_identity = identity("a", "b", "c", b"scope")
  let pages = physical_pages(4)
  let left_first = publish_prefix(
    left,
    cache_identity,
    [1, 2, 3, 4],
    pages[0:2],
  )
  let right_first = publish_prefix(
    right,
    cache_identity,
    [1, 2, 3, 4],
    pages[0:2],
  )
  ignore(publish_prefix(left, cache_identity, [1, 2, 5, 6], pages[2:4]))
  ignore(publish_prefix(right, cache_identity, [1, 2, 5, 6], pages[2:4]))
  left.retain_active(left_first)
  right.retain_active(right_first)
  left.release_active(left_first)
  right.release_active(right_first)
  let left_eviction = @radix.PrefixEvictionBuffer::new(4)
  let right_eviction = @radix.PrefixEvictionBuffer::new(4)
  assert_eq(
    evict_prefix(left, left_eviction),
    evict_prefix(right, right_eviction),
  )
  assert_eq(left.debug_snapshot(), right.debug_snapshot())
  left.debug_check_invariants()
  right.debug_check_invariants()
}
