///|
fn Scheduler::evict_one_prefix(self : Scheduler) -> Int {
  if !self.capacity.prefix_enabled() {
    return 0
  }
  let before = self.page_allocator.free_page_count()
  let output = self.prefix_eviction.unwrap()
  let start = try self.prefix_index.unwrap().try_plan_eviction(output) catch {
    _ => return 0
  } noraise {
    value => value
  }
  if !start.is_planned() {
    return 0
  }
  let plan = try! start.plan()
  for index in 0..<output.page_count() {
    if (try! self.page_allocator.cached_reference_count(
        try! output.page_at(index),
      )) <=
      0 {
      abort("prefix cached-reference preflight invariant")
    }
  }
  ignore(try! self.prefix_index.unwrap().commit_eviction(plan, output))
  for index in 0..<output.page_count() {
    try! self.page_allocator.release_cached(try! output.page_at(index))
  }
  self.prefix_evictions = Scheduler::prefix_saturating_add(
    self.prefix_evictions,
    1,
  )
  self.page_allocator.free_page_count() - before
}

///|
fn Scheduler::evict_for_virtual_room(
  self : Scheduler,
  virtual_free : Int,
  required : Int,
) -> Int {
  if !self.capacity.prefix_enabled() {
    return virtual_free
  }
  let mut available = virtual_free
  while available < required {
    let entries = self.prefix_index.unwrap().entry_count()
    if entries == 0 {
      break
    }
    let freed = self.evict_one_prefix()
    if freed == 0 && self.prefix_index.unwrap().entry_count() == entries {
      break
    }
    available += freed
  }
  available
}

///|
fn Scheduler::prefix_publish_preflight(
  self : Scheduler,
  pages : @radix.PrefixPublishBuffer,
) -> Bool {
  let maximum = self.page_allocator.limits().max_cached_references_per_page()
  for index in 0..<pages.length() {
    let page = try! pages.page_at(index)
    if (try! pages.planned_adoption(index)) &&
      (try! self.page_allocator.cached_reference_count(page)) >= maximum {
      return false
    }
  }
  true
}

///|
fn Scheduler::commit_prompt_prefix_plan(
  self : Scheduler,
  index : @radix.PrefixIndex,
  tokens : @radix.PrefixTokenBuffer,
  pages : @radix.PrefixPublishBuffer,
  priority : Int,
  plan : @radix.PrefixPublishPlan,
) -> Unit {
  if !self.prefix_publish_preflight(pages) {
    try! index.abort_publish(plan)
    return
  }
  for page_index in 0..<pages.length() {
    if (try! pages.planned_adoption(page_index)) {
      try! self.page_allocator.retain_cached(try! pages.page_at(page_index))
    }
  }
  let published = try index.publish(plan, tokens, pages, priority~) catch {
    _ => {
      try! index.abort_publish(plan)
      false
    }
  } noraise {
    _ => true
  }
  for page_index in 0..<pages.length() {
    let planned = try! pages.planned_adoption(page_index)
    let adopted = try! pages.was_adopted(page_index)
    if published && planned != adopted {
      abort("prefix publication adoption invariant")
    }
    if planned && !published {
      try! self.page_allocator.release_cached(try! pages.page_at(page_index))
    }
  }
  if published {
    self.prefix_publications = Scheduler::prefix_saturating_add(
      self.prefix_publications,
      1,
    )
  }
}

///|
fn Scheduler::publish_prompt_prefix(self : Scheduler, slot : Int) -> Unit {
  if !self.capacity.prefix_enabled() {
    return
  }
  let request = self.requests[slot].unwrap()
  if !request.has_prefix_key() {
    return
  }
  let key = try! request.prefix_key(self.capacity.prefix_layout())
  if (try! key.cache_permission()) != ReadWrite {
    return
  }
  let token_limit = self.prefix_index.unwrap().limits().max_tokens_per_entry()
  let page_size = self.capacity.tokens_per_page()
  let token_count = Int::min(request.input_tokens().length(), token_limit) /
    page_size *
    page_size
  if token_count == 0 {
    return
  }
  let page_count = token_count / page_size
  let tokens = self.prefix_tokens.unwrap()
  tokens.clear()
  for index in 0..<token_count {
    try! tokens.push_token(try! request.input_token_at(index))
  }
  let pages = self.prefix_pages.unwrap()
  pages.clear()
  let table = try! self.request_tables.read(slot)
  for index in 0..<page_count {
    try! pages.push_page(try! self.block_tables.page_at(table, index))
  }
  let index = self.prefix_index.unwrap()
  let mut attempts = 0
  while attempts <= index.limits().entry_capacity() {
    attempts += 1
    try index.try_plan_publish(key, tokens, pages, priority=page_count) catch {
      _ => abort("scheduler prefix publication preflight invariant")
    } noraise {
      start => {
        if start.is_duplicate() {
          try! index.touch(try! start.duplicate_entry())
          return
        }
        if start.is_capacity_exhausted() {
          let before = index.entry_count()
          ignore(self.evict_one_prefix())
          if index.entry_count() == before {
            return
          }
          continue
        }
        if start.is_rejected() {
          return
        }
        let plan = try! start.plan()
        self.commit_prompt_prefix_plan(index, tokens, pages, page_count, plan)
        return
      }
    }
  }
}
