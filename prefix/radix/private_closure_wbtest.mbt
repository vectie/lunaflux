///|
fn wb_closure_inference_limits() -> @inference.InferenceLimits raise {
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
struct ClosureIdentity {
  model : @spec.ModelIdentity
  tokenizer : @tokenizer.TokenizerDigest
  layout : PrefixLayout
}

///|
fn wb_closure_identity() -> ClosureIdentity raise {
  {
    model: @spec.ModelIdentity::new(
      @spec.ContentDigest::from_sha256("a".repeat(64)),
      @spec.PlanDigest::from_sha256("b".repeat(64)),
    ),
    tokenizer: @tokenizer.TokenizerDigest::parse("c".repeat(64)),
    layout: PrefixLayout::new(version=1, tokens_per_page=2),
  }
}

///|
fn wb_closure_index() -> PrefixIndex raise {
  PrefixIndex::new(
    PrefixIndexLimits::new(
      entry_capacity=8,
      node_capacity=32,
      token_capacity=32,
      max_tokens_per_entry=16,
      page_capacity=16,
      max_scope_bytes=16,
      max_active_references_per_entry=2,
      max_generation=100,
    ),
  )
}

///|
fn wb_closure_page() -> @page_allocator.PageId raise {
  let allocator = @page_allocator.PageAllocator::new(
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=1,
      max_run_pages=1,
      max_active_references_per_page=8,
      max_cached_references_per_page=8,
    ),
  )
  allocator.allocate_one()
}

///|
fn wb_closure_request_key(
  identity : ClosureIdentity,
) -> (PrefixRequestKey, @inference.LunaRequestSemanticLease) raise {
  let limits = wb_closure_inference_limits()
  let storage = @inference.LunaRequestSemanticStorage::new(
    limits,
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
fn wb_closure_publish(
  index : PrefixIndex,
  identity : ClosureIdentity,
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
  let (key, lease) = wb_closure_request_key(identity)
  let start = index.try_plan_publish(key, tokens, pages, priority=1)
  if !start.is_planned() {
    lease.release()
    abort("expected closure publication plan")
  }
  let entry = index.publish(start.plan(), tokens, pages, priority=1)
  lease.release()
  entry
}

///|
fn wb_closure_lookup(
  index : PrefixIndex,
  identity : ClosureIdentity,
  values : ArrayView[Int],
  output : PrefixLookupBuffer,
) -> Unit raise {
  let tokens = PrefixTokenBuffer::new(values.length())
  for token in values {
    tokens.push_token(token)
  }
  let (key, lease) = wb_closure_request_key(identity)
  index.lookup(key, tokens, maximum_reusable_tokens=values.length(), output)
  lease.release()
}

///|
test "terminal publication generation disables mutation without harming lookup" {
  let index = wb_closure_index()
  let identity = wb_closure_identity()
  let page = wb_closure_page()
  ignore(wb_closure_publish(index, identity, [1, 2], [page]))
  let (key, lease) = wb_closure_request_key(identity)
  let tokens = PrefixTokenBuffer::new(2)
  tokens.push_token(3)
  tokens.push_token(4)
  let pages = PrefixPublishBuffer::new(1)
  pages.push_page(page)
  index.publish_plan_generation = 0xffffffffffffffffUL
  assert_true(
    index.try_plan_publish(key, tokens, pages, priority=1).is_rejected(),
  )
  let lookup = PrefixLookupBuffer::new(1)
  wb_closure_lookup(index, identity, [1, 2], lookup)
  assert_eq(lookup.page_at(0), page)
  lease.release()
  index.debug_check_invariants()
}

///|
test "terminal eviction generation disables mutation without harming lookup" {
  let index = wb_closure_index()
  let identity = wb_closure_identity()
  let page = wb_closure_page()
  ignore(wb_closure_publish(index, identity, [1, 2], [page]))
  let output = PrefixEvictionBuffer::new(1)
  index.eviction_plan_generation = 0xffffffffffffffffUL
  assert_true(index.try_plan_eviction(output).is_rejected())
  let lookup = PrefixLookupBuffer::new(1)
  wb_closure_lookup(index, identity, [1, 2], lookup)
  assert_eq(lookup.page_at(0), page)
  index.debug_check_invariants()
}
