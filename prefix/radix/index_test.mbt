///|
fn inference_limits() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=64,
    max_input_tokens=64,
    max_new_tokens=16,
    max_context_tokens=64,
    max_token_id=1023,
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
fn identity(
  content_digit : String,
  plan_digit : String,
  tokenizer_digit : String,
  scope : Bytes,
  page_size? : Int = 2,
) -> @radix.PrefixIdentity raise {
  @radix.PrefixIdentity::new(
    model_identity=@spec.ModelIdentity::new(
      @spec.ContentDigest::from_sha256(content_digit.repeat(64)),
      @spec.PlanDigest::from_sha256(plan_digit.repeat(64)),
    ),
    tokenizer=@tokenizer.TokenizerDigest::parse(tokenizer_digit.repeat(64)),
    scope=@inference.CacheScope::from_ascii(scope, inference_limits()),
    layout=@radix.PrefixLayout::new(version=1, tokens_per_page=page_size),
  )
}

///|
fn index_limits(
  entries? : Int = 8,
  nodes? : Int = 32,
  tokens? : Int = 16,
  pages? : Int = 16,
  active? : Int = 2,
  generation? : Int = 100,
) -> @radix.PrefixIndexLimits raise {
  @radix.PrefixIndexLimits::new(
    entry_capacity=entries,
    node_capacity=nodes,
    max_tokens_per_entry=tokens,
    page_capacity=pages,
    max_active_references_per_entry=active,
    max_generation=generation,
  )
}

///|
fn physical_pages(count : Int) -> Array[@page_allocator.PageId] raise {
  let allocator = @page_allocator.PageAllocator::new(
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=count,
      max_run_pages=count,
      max_active_references_per_page=8,
      max_cached_references_per_page=8,
    ),
  )
  let pages : Array[@page_allocator.PageId] = []
  for _ in 0..<count {
    pages.push(allocator.allocate_one())
  }
  pages
}

///|
test "longest lookup returns only complete pages and copied identities" {
  let index = @radix.PrefixIndex::new(index_limits())
  let cache_identity = identity("a", "b", "c", b"scope-a")
  let pages = physical_pages(4)
  let short = index.insert(cache_identity, [1, 2], pages[0:1])
  let long = index.insert(cache_identity, [1, 2, 3, 4], pages[1:3])
  let result = @radix.PrefixLookupBuffer::new(4)
  index.lookup_longest_full_pages(cache_identity, [1, 2, 3, 4, 99], result)
  assert_eq(result.entry(), Some(long))
  assert_eq(result.matched_tokens(), 4)
  assert_eq(result.page_count(), 2)
  assert_eq(result.page_at(0), pages[1])
  assert_eq(result.page_at(1), pages[2])
  index.lookup_longest_full_pages(cache_identity, [1, 2, 3], result)
  assert_eq(result.entry(), Some(short))
  assert_eq(result.matched_tokens(), 2)
  index.debug_check_invariants()
}

///|
test "model tokenizer scope and layout identities are isolated" {
  let index = @radix.PrefixIndex::new(index_limits())
  let base = identity("a", "b", "c", b"scope-a")
  let pages = physical_pages(1)
  ignore(index.insert(base, [7, 8], pages))
  let result = @radix.PrefixLookupBuffer::new(2)
  let variants = [
    identity("d", "b", "c", b"scope-a"),
    identity("a", "e", "c", b"scope-a"),
    identity("a", "b", "f", b"scope-a"),
    identity("a", "b", "c", b"scope-b"),
    identity("a", "b", "c", b"scope-a", page_size=1),
  ]
  for variant in variants {
    index.lookup_longest_full_pages(variant, [7, 8], result)
    assert_eq(result.page_count(), 0)
    assert_eq(result.entry(), None)
  }
}

///|
test "insert preflight preserves state on shape and capacity failure" {
  let index = @radix.PrefixIndex::new(
    index_limits(entries=2, nodes=3, tokens=8, pages=2),
  )
  let cache_identity = identity("a", "b", "c", b"scope")
  let pages = physical_pages(3)
  try index.insert(cache_identity, [1, 2, 3], pages[0:1]) catch {
    error => debug_inspect(error, content="InvalidPageCount")
  } noraise {
    _ => fail("expected partial-page rejection")
  }
  assert_eq(index.entry_count(), 0)
  let first = index.insert(cache_identity, [1, 2], pages[0:1])
  let before = index.debug_snapshot()
  try index.insert(cache_identity, [3, 4, 5, 6], pages[1:3]) catch {
    error => debug_inspect(error, content="Exhausted")
  } noraise {
    _ => fail("expected node exhaustion")
  }
  assert_eq(index.debug_snapshot(), before)
  try index.insert(cache_identity, [1, 2], pages[1:2]) catch {
    DuplicatePrefix(entry) => assert_eq(entry, first)
    _ => fail("unexpected duplicate error")
  } noraise {
    _ => fail("expected duplicate rejection")
  }
  index.debug_check_invariants()
}

///|
test "active references guard deterministic eviction and stale generations" {
  let index = @radix.PrefixIndex::new(
    index_limits(entries=2, nodes=8, pages=2, active=1),
  )
  let cache_identity = identity("a", "b", "c", b"scope")
  let pages = physical_pages(2)
  let first = index.insert(cache_identity, [1, 2], pages[0:1])
  let second = index.insert(cache_identity, [3, 4], pages[1:2])
  let lookup = @radix.PrefixLookupBuffer::new(1)
  index.lookup_longest_full_pages(cache_identity, [1, 2], lookup)
  index.retain_active(first)
  assert_eq(index.active_reference_count(first), 1)
  try index.retain_active(first) catch {
    ReferenceOverflow(entry) => assert_eq(entry, first)
    _ => fail("unexpected reference error")
  } noraise {
    _ => fail("expected reference overflow")
  }
  let evicted_pages = @radix.PrefixEvictionBuffer::new(1)
  assert_eq(index.evict_oldest_zero_reference(evicted_pages), second)
  assert_eq(evicted_pages.page_at(0), pages[1])
  try index.retain_active(second) catch {
    StaleEntry(entry) => assert_eq(entry, second)
    _ => fail("unexpected stale-entry error")
  } noraise {
    _ => fail("expected stale entry")
  }
  index.release_active(first)
  try index.release_active(first) catch {
    ReferenceUnderflow(entry) => assert_eq(entry, first)
    _ => fail("unexpected reference error")
  } noraise {
    _ => fail("expected reference underflow")
  }
  assert_eq(index.evict_oldest_zero_reference(evicted_pages), first)
  index.debug_check_invariants()
}

///|
test "insert copies caller tokens and page run" {
  let index = @radix.PrefixIndex::new(index_limits())
  let cache_identity = identity("a", "b", "c", b"scope")
  let tokens = [11, 12]
  let pages = physical_pages(2)
  let original = pages[0]
  ignore(index.insert(cache_identity, tokens, pages[0:1]))
  tokens[0] = 99
  pages[0] = pages[1]
  let result = @radix.PrefixLookupBuffer::new(1)
  index.lookup_longest_full_pages(cache_identity, [11, 12], result)
  assert_eq(result.page_at(0), original)
  index.lookup_longest_full_pages(cache_identity, [99, 12], result)
  assert_eq(result.page_count(), 0)
}

///|
test "too-small result buffers preserve prior output and index state" {
  let index = @radix.PrefixIndex::new(index_limits())
  let cache_identity = identity("a", "b", "c", b"scope")
  let pages = physical_pages(3)
  let short = index.insert(cache_identity, [1, 2], pages[0:1])
  ignore(index.insert(cache_identity, [1, 2, 3, 4], pages[1:3]))
  let result = @radix.PrefixLookupBuffer::new(1)
  index.lookup_longest_full_pages(cache_identity, [1, 2], result)
  let before_index = index.debug_snapshot()
  assert_eq(result.entry(), Some(short))
  assert_eq(result.page_at(0), pages[0])
  try
    index.lookup_longest_full_pages(cache_identity, [1, 2, 3, 4], result)
  catch {
    error => debug_inspect(error, content="ResultBufferTooSmall")
  } noraise {
    _ => fail("expected lookup buffer rejection")
  }
  assert_eq(index.debug_snapshot(), before_index)
  assert_eq(result.entry(), Some(short))
  assert_eq(result.page_at(0), pages[0])

  index.retain_active(short)
  let eviction = @radix.PrefixEvictionBuffer::new(1)
  let before_eviction = index.debug_snapshot()
  try index.evict_oldest_zero_reference(eviction) catch {
    error => debug_inspect(error, content="ResultBufferTooSmall")
  } noraise {
    _ => fail("expected eviction buffer rejection")
  }
  assert_eq(index.debug_snapshot(), before_eviction)
  assert_eq(eviction.entry(), None)
  assert_eq(eviction.page_count(), 0)
}

///|
test "invalid tokens fail closed without mutating prior lookup output" {
  let index = @radix.PrefixIndex::new(index_limits())
  let cache_identity = identity("a", "b", "c", b"scope")
  let pages = physical_pages(1)
  let entry = index.insert(cache_identity, [1, 2], pages)
  let result = @radix.PrefixLookupBuffer::new(1)
  index.lookup_longest_full_pages(cache_identity, [1, 2], result)
  let before = index.debug_snapshot()
  try index.lookup_longest_full_pages(cache_identity, [1, -2], result) catch {
    InvalidToken(index) => assert_eq(index, 1)
    _ => fail("unexpected invalid-token error")
  } noraise {
    _ => fail("expected invalid-token rejection")
  }
  assert_eq(index.debug_snapshot(), before)
  assert_eq(result.entry(), Some(entry))
  assert_eq(result.page_at(0), pages[0])
  try index.insert(cache_identity, [3, -4], pages) catch {
    InvalidToken(index) => assert_eq(index, 1)
    _ => fail("unexpected invalid-token error")
  } noraise {
    _ => fail("expected invalid-token rejection")
  }
  assert_eq(index.debug_snapshot(), before)
}

///|
test "deterministic pseudo-random churn preserves invariants" {
  let index = @radix.PrefixIndex::new(
    index_limits(entries=4, nodes=12, tokens=8, pages=4, generation=100),
  )
  let cache_identity = identity("a", "b", "c", b"scope")
  let pages = physical_pages(1)
  let eviction = @radix.PrefixEvictionBuffer::new(1)
  let mut state = 17
  for _ in 0..<50 {
    state = (state * 109 + 89) % 997
    let entry = index.insert(cache_identity, [state, state + 1], pages)
    if state % 3 == 0 {
      index.retain_active(entry)
      index.release_active(entry)
    }
    assert_eq(index.evict_oldest_zero_reference(eviction), entry)
    index.debug_check_invariants()
  }
  assert_eq(index.entry_count(), 0)
  assert_eq(index.page_count(), 0)
}
