///|
fn complexity_limits(
  entries : Int,
  nodes : Int,
  tokens : Int,
  pages : Int,
) -> PrefixIndexLimits raise {
  PrefixIndexLimits::new(
    entry_capacity=entries,
    node_capacity=nodes,
    token_capacity=tokens,
    max_tokens_per_entry=tokens,
    page_capacity=pages,
    max_scope_bytes=16,
    max_active_references_per_entry=entries,
    max_generation=100,
  )
}

///|
fn complexity_inference_limits() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=64,
    max_input_tokens=256,
    max_new_tokens=16,
    max_context_tokens=256,
    max_token_id=4095,
    max_stop_token_ids=4,
    max_stop_strings=4,
    max_stop_string_bytes=16,
    max_trace_bytes=16,
    max_cache_scope_bytes=16,
    max_decoded_delta_bytes=16,
    max_deadline_millis=1000UL,
    max_top_k=128,
    max_temperature=4.0,
  )
}

///|
struct ComplexityIdentity {
  model : @spec.ModelIdentity
  tokenizer : @tokenizer.TokenizerDigest
  layout : PrefixLayout
}

///|
fn complexity_identity(digit : String) -> ComplexityIdentity raise {
  {
    model: @spec.ModelIdentity::new(
      @spec.ContentDigest::from_sha256(digit.repeat(64)),
      @spec.PlanDigest::from_sha256("f".repeat(64)),
    ),
    tokenizer: @tokenizer.TokenizerDigest::parse("e".repeat(64)),
    layout: PrefixLayout::new(version=1, tokens_per_page=2),
  }
}

///|
fn complexity_key(
  identity : ComplexityIdentity,
) -> (PrefixRequestKey, @inference.LunaRequestSemanticLease) raise {
  let storage = @inference.LunaRequestSemanticStorage::new(
    complexity_inference_limits(),
    @inference.LunaRequestSemanticStepBudget::new(17),
  )
  let write = storage.begin(
    stop_token_count=0,
    stop_string_count=0,
    cache_scope_byte_count=5,
    cache_permission=ReadWrite,
  )
  for byte in b"scope" {
    write.push_cache_scope_byte(byte)
  }
  let work = write.finish()
  while work.progress().is_pending() {

  }
  let lease = work.take_lease()
  (
    PrefixRequestKey::new(
      model_identity=identity.model,
      tokenizer=identity.tokenizer,
      layout=identity.layout,
      semantic=lease.prefix_view(),
    ),
    lease,
  )
}

///|
fn complexity_publish(
  index : PrefixIndex,
  identity : ComplexityIdentity,
  values : ArrayView[Int],
  physical : ArrayView[@page_allocator.PageId],
) -> PrefixEntryId raise {
  let tokens = PrefixTokenBuffer::new(values.length())
  for token in values {
    tokens.push_token(token)
  }
  let pages = PrefixPublishBuffer::new(physical.length())
  for page in physical {
    pages.push_page(page)
  }
  let (key, lease) = complexity_key(identity)
  let start = index.try_plan_publish(
    key,
    tokens,
    pages,
    priority=physical.length(),
  )
  if !start.is_planned() {
    lease.release()
    abort("expected complexity publication plan")
  }
  let entry = index.publish(
    start.plan(),
    tokens,
    pages,
    priority=physical.length(),
  )
  lease.release()
  entry
}

///|
fn complexity_lookup(
  index : PrefixIndex,
  identity : ComplexityIdentity,
  values : ArrayView[Int],
  output : PrefixLookupBuffer,
) -> Unit raise {
  let tokens = PrefixTokenBuffer::new(values.length())
  for token in values {
    tokens.push_token(token)
  }
  let (key, lease) = complexity_key(identity)
  index.lookup(key, tokens, maximum_reusable_tokens=values.length(), output)
  lease.release()
}

///|
fn complexity_evict(
  index : PrefixIndex,
  output : PrefixEvictionBuffer,
) -> PrefixEntryId raise {
  let start = index.try_plan_eviction(output)
  if !start.is_planned() {
    abort("expected complexity eviction plan")
  }
  index.commit_eviction(start.plan(), output)
}

///|
fn complexity_pages(count : Int) -> Array[@page_allocator.PageId] raise {
  let allocator = @page_allocator.PageAllocator::new(
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=count,
      max_run_pages=count,
      max_active_references_per_page=count,
      max_cached_references_per_page=count,
    ),
  )
  let pages : Array[@page_allocator.PageId] = []
  for _ in 0..<count {
    pages.push(allocator.allocate_one())
  }
  pages
}

///|
test "root and child lookup counters stay logarithmic under adversarial order" {
  let digits = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f",
  ]
  let pages = complexity_pages(16)
  let roots = PrefixIndex::new(complexity_limits(16, 16, 32, 16))
  for index in 0..<16 {
    ignore(
      complexity_publish(
        roots,
        complexity_identity(digits[index]),
        [index * 2, index * 2 + 1],
        pages[index:index + 1],
      ),
    )
  }
  roots.root_lookup_comparisons = 0
  let output = PrefixLookupBuffer::new(1)
  for index in 0..<16 {
    complexity_lookup(
      roots,
      complexity_identity(digits[index]),
      [index * 2, index * 2 + 1],
      output,
    )
    assert_eq(output.page_count(), 1)
  }
  assert_true(roots.root_lookup_comparisons >= 16)
  assert_true(roots.root_lookup_comparisons <= 80)
  roots.debug_check_invariants()

  let branches = PrefixIndex::new(complexity_limits(16, 16, 32, 16))
  let shared = complexity_identity("a")
  for index in 0..<16 {
    ignore(
      complexity_publish(
        branches,
        shared,
        [index * 2, index * 2 + 1],
        pages[index:index + 1],
      ),
    )
  }
  branches.child_lookup_comparisons = 0
  for index in 0..<16 {
    complexity_lookup(branches, shared, [index * 2, index * 2 + 1], output)
    assert_eq(output.page_count(), 1)
  }
  assert_true(branches.child_lookup_comparisons >= 16)
  assert_true(branches.child_lookup_comparisons <= 80)
  branches.debug_check_invariants()
}

///|
test "duplicate-page heapsort has a structural n-log-n bound" {
  let count = 64
  let index = PrefixIndex::new(complexity_limits(1, 1, count * 2, count))
  let physical = complexity_pages(count)
  let tokens = PrefixTokenBuffer::new(count * 2)
  let pages = PrefixPublishBuffer::new(count)
  for token in 0..<(count * 2) {
    tokens.push_token(token)
  }
  for page in 0..<(count - 1) {
    pages.push_page(physical[page])
  }
  pages.push_page(physical[0])
  try index.validate_publish_shape(tokens, pages, 2, 1) catch {
    InvalidPageCount => ()
    _ => fail("unexpected duplicate-page validation error")
  } noraise {
    _ => fail("duplicate physical page was accepted")
  }
  assert_eq(index.entry_count(), 0)
  assert_true(index.duplicate_page_comparisons <= count * 14)
  assert_true(index.duplicate_page_swaps <= count * 8)
  for page in 0..<count {
    assert_false(try! index.duplicate_page_scratch.is_occupied(page))
  }
  index.debug_check_invariants()
}

///|
test "recency exhaustion normalization is deterministic n-log-n work" {
  let empty = PrefixIndex::new(complexity_limits(1, 1, 2, 1))
  empty.recency_normalization_comparisons = 17
  empty.recency_normalization_swaps = 19
  empty.normalize_recencies()
  assert_eq(empty.recency_normalization_comparisons, 0)
  assert_eq(empty.recency_normalization_swaps, 0)
  empty.debug_check_invariants()

  let count = 64
  let index = PrefixIndex::new(
    complexity_limits(count, count, count * 2, count),
  )
  let identity = complexity_identity("a")
  let pages = complexity_pages(count)
  let entries : Array[PrefixEntryId] = []
  for entry in 0..<count {
    entries.push(
      complexity_publish(
        index,
        identity,
        [entry * 2, entry * 2 + 1],
        pages[entry:entry + 1],
      ),
    )
  }
  for entry in 0..<count {
    index.entry_recencies[entries[entry].index()] = (count - entry).to_uint64()
  }
  index.eviction_heap_rebuild()
  index.recency_normalization_comparisons = 1_000_000
  index.recency_normalization_swaps = 1_000_000
  index.recency_clock = 0xffffffffffffffffUL
  index.touch(entries[0])
  assert_true(index.recency_normalization_comparisons > 0)
  assert_true(index.recency_normalization_comparisons <= count * 20)
  assert_true(index.recency_normalization_swaps > 0)
  assert_true(index.recency_normalization_swaps <= count * 10)
  assert_eq(index.entry_recencies[entries[0].index()], 65UL)
  let eviction = PrefixEvictionBuffer::new(1)
  assert_eq(complexity_evict(index, eviction), entries[count - 1])
  index.debug_check_invariants()
}

///|
test "arena free stacks and zero-reference heap have local bounds" {
  let index = PrefixIndex::new(complexity_limits(3, 4, 6, 3))
  let identity = complexity_identity("a")
  let pages = complexity_pages(3)
  let first = complexity_publish(index, identity, [1, 2], pages[0:1])
  assert_eq(index.free_stack_touches, 8)
  ignore(complexity_publish(index, identity, [3, 4], pages[1:2]))
  ignore(complexity_publish(index, identity, [5, 6], pages[2:3]))
  index.eviction_heap_touches = 0
  index.touch(first)
  assert_true(index.eviction_heap_touches <= 8)
  index.free_stack_touches = 0
  let eviction = PrefixEvictionBuffer::new(1)
  ignore(complexity_evict(index, eviction))
  assert_eq(index.free_stack_touches, 6)
  index.debug_check_invariants()
}

///|
test "eviction and compaction visit only changed path and incident edges" {
  let count = 16
  let index = PrefixIndex::new(
    complexity_limits(count, count, count * 2, count),
  )
  let identity = complexity_identity("a")
  let pages = complexity_pages(count)
  let entries : Array[PrefixEntryId] = []
  let tokens : Array[Int] = []
  for depth in 1..<(count + 1) {
    tokens.push(depth * 2 - 1)
    tokens.push(depth * 2)
    entries.push(complexity_publish(index, identity, tokens[:], pages[0:depth]))
  }
  for entry in 0..<(count - 1) {
    index.retain_active(entries[entry])
  }
  index.compaction_node_visits = 0
  index.compaction_incident_edges = 0
  let eviction = PrefixEvictionBuffer::new(count)
  assert_eq(complexity_evict(index, eviction), entries[count - 1])
  assert_true(index.compaction_node_visits <= count * 2)
  assert_true(index.compaction_incident_edges <= count)
  index.debug_check_invariants()
}
