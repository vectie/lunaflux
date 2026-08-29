///|
fn PrefixIndex::has_new_root_room(
  self : PrefixIndex,
  token_count : Int,
  page_count : Int,
) -> Bool {
  self.free_root() >= 0 &&
  self.free_entry() >= 0 &&
  self.free_node_count() >= 1 &&
  self.free_token_count() >= token_count &&
  self.free_page_count() >= page_count
}

///|
fn PrefixIndex::preflight_new_root(
  self : PrefixIndex,
  token_count : Int,
  page_count : Int,
) -> Unit raise PrefixIndexError {
  if !self.has_new_root_room(token_count, page_count) {
    raise Exhausted
  }
}

///|
fn PrefixIndex::install_request_root(
  self : PrefixIndex,
  key : PrefixRequestKey,
) -> Int raise PrefixIndexError {
  if !key.is_live() {
    raise StaleKey
  }
  let length = key.semantic.scope_length() catch { _ => raise StaleKey }
  if length > self.limits.max_scope_bytes() {
    raise InvalidLimit(ScopeBytes)
  }
  for index in 0..<length {
    self.root_scope_scratch[index] = key.semantic.scope_byte_at(index) catch {
      _ => {
        for clear in 0..<index {
          self.root_scope_scratch[clear] = b'\x00'
        }
        raise StaleKey
      }
    }
  }
  let root = self.take_free_root()
  if root < 0 {
    for index in 0..<length {
      self.root_scope_scratch[index] = b'\x00'
    }
    raise Exhausted
  }
  self.root_used[root] = true
  self.root_models[root] = Some(key.model_identity)
  self.root_tokenizers[root] = Some(key.tokenizer)
  self.root_layouts[root] = Some(key.layout)
  self.root_scope_lengths[root] = length
  for index in 0..<length {
    self.root_scope_bytes[root * self.limits.max_scope_bytes() + index] = self.root_scope_scratch[index]
    self.root_scope_scratch[index] = b'\x00'
  }
  self.root_child_heads[root] = -1
  self.root_child_counts[root] = 0
  self.root_index_insert(root)
  root
}

///|
fn PrefixIndex::require_publish_plan(
  self : PrefixIndex,
  plan : PrefixPublishPlan,
) -> Unit raise PrefixIndexError {
  if !physical_equal(plan.owner, self) ||
    !self.publish_plan_active ||
    plan.generation != self.publish_plan_generation {
    raise StalePublishPlan
  }
}

///|
fn PrefixIndex::publish_plan_matches(
  self : PrefixIndex,
  key : PrefixRequestKey,
  tokens : PrefixTokenBuffer,
  pages : PrefixPublishBuffer,
  priority : Int,
) -> Bool raise PrefixIndexError {
  if tokens.length() != self.publish_plan_token_count ||
    pages.length() != self.publish_plan_page_count ||
    priority != self.publish_plan_priority ||
    self.find_request_root(key) != self.publish_plan_root {
    return false
  }
  for index in 0..<tokens.length() {
    if tokens.token_at(index) != self.publish_token_scratch[index] {
      return false
    }
  }
  for index in 0..<pages.length() {
    let planned_page = try! self.publish_page_scratch.read(index)
    if pages.page_at(index) != planned_page {
      return false
    }
  }
  self.mark_expected_adoptions(
    self.publish_plan_root,
    tokens,
    key.layout.tokens_per_page(),
    pages.adopted,
    pages.length(),
  )
  for index in 0..<pages.length() {
    if pages.planned[index] != pages.adopted[index] {
      return false
    }
  }
  true
}

///|
fn PrefixIndex::clear_publish_plan(self : PrefixIndex) -> Unit {
  for index in 0..<self.publish_plan_token_count {
    self.publish_token_scratch[index] = 0
  }
  for index in 0..<self.publish_plan_page_count {
    try! self.publish_page_scratch.clear(index)
  }
  self.publish_plan_active = false
  self.publish_plan_root = -1
  self.publish_plan_token_count = 0
  self.publish_plan_page_count = 0
  self.publish_plan_priority = 0
}

///|
pub fn PrefixPublishStart::is_planned(self : PrefixPublishStart) -> Bool {
  self.state == 0
}

///|
pub fn PrefixPublishStart::is_duplicate(self : PrefixPublishStart) -> Bool {
  self.state == 1
}

///|
pub fn PrefixPublishStart::is_capacity_exhausted(
  self : PrefixPublishStart,
) -> Bool {
  self.state == 2
}

///|
pub fn PrefixPublishStart::is_rejected(self : PrefixPublishStart) -> Bool {
  self.state >= 3
}

///|
pub fn PrefixPublishStart::plan(
  self : PrefixPublishStart,
) -> PrefixPublishPlan raise PrefixIndexError {
  if self.state != 0 {
    raise StalePublishPlan
  }
  { owner: self.owner, generation: self.generation, key: self.key }
}

///|
pub fn PrefixPublishStart::duplicate_entry(
  self : PrefixPublishStart,
) -> PrefixEntryId raise PrefixIndexError {
  if self.state != 1 {
    raise StalePublishPlan
  }
  PrefixEntryId(self.entry_index, self.entry_generation)
}

///|
/// Preflights the exact compressed insertion without constructing an error for
/// ordinary duplicate, cache-capacity, scope-envelope, or terminal-generation
/// outcomes. A planned result snapshots the exact buffers and key until
/// `publish` or `abort_publish` consumes the extracted plan.
pub fn PrefixIndex::try_plan_publish(
  self : PrefixIndex,
  key : PrefixRequestKey,
  tokens : PrefixTokenBuffer,
  pages : PrefixPublishBuffer,
  priority~ : Int,
) -> PrefixPublishStart raise PrefixIndexError {
  if self.publish_plan_active || self.eviction_plan_active {
    raise PublishPlanBusy
  }
  if self.publish_plan_generation == 0xffffffffffffffffUL {
    return {
      state: 4,
      owner: self,
      generation: self.publish_plan_generation,
      key,
      entry_index: -1,
      entry_generation: 0,
    }
  }
  if !key.is_live() {
    raise StaleKey
  }
  if key.cache_permission() != ReadWrite {
    raise CachePermissionDenied
  }
  let scope_length = key.semantic.scope_length() catch { _ => raise StaleKey }
  if scope_length > self.limits.max_scope_bytes() {
    return {
      state: 3,
      owner: self,
      generation: self.publish_plan_generation,
      key,
      entry_index: -1,
      entry_generation: 0,
    }
  }
  let page_size = key.layout.tokens_per_page()
  self.validate_publish_shape(tokens, pages, page_size, priority)
  let root = self.find_request_root(key)
  let preflight : PrefixExistingPublishPreflight = if root < 0 {
    if self.has_new_root_room(tokens.length(), pages.length()) {
      { state: 0, duplicate_entry: -1 }
    } else {
      { state: 2, duplicate_entry: -1 }
    }
  } else {
    self.existing_publish_preflight(root, tokens, page_size)
  }
  if preflight.state == 1 {
    let duplicate_entry = preflight.duplicate_entry
    return {
      state: 1,
      owner: self,
      generation: self.publish_plan_generation,
      key,
      entry_index: duplicate_entry,
      entry_generation: self.entry_generations[duplicate_entry],
    }
  }
  if preflight.state == 2 {
    return {
      state: 2,
      owner: self,
      generation: self.publish_plan_generation,
      key,
      entry_index: -1,
      entry_generation: 0,
    }
  }
  self.mark_expected_adoptions(
    root,
    tokens,
    page_size,
    pages.planned,
    pages.length(),
  )
  for index in 0..<tokens.length() {
    self.publish_token_scratch[index] = tokens.token_at(index)
  }
  for index in 0..<pages.length() {
    try! self.publish_page_scratch.write(index, try! pages.page_at(index))
  }
  self.publish_plan_generation += 1UL
  self.publish_plan_active = true
  self.publish_plan_root = root
  self.publish_plan_token_count = tokens.length()
  self.publish_plan_page_count = pages.length()
  self.publish_plan_priority = priority
  {
    state: 0,
    owner: self,
    generation: self.publish_plan_generation,
    key,
    entry_index: -1,
    entry_generation: 0,
  }
}

///|
pub fn PrefixIndex::publish(
  self : PrefixIndex,
  plan : PrefixPublishPlan,
  tokens : PrefixTokenBuffer,
  pages : PrefixPublishBuffer,
  priority~ : Int,
) -> PrefixEntryId raise PrefixIndexError {
  self.require_publish_plan(plan)
  let key = plan.key
  if !key.is_live() {
    raise StaleKey
  }
  let page_size = key.layout.tokens_per_page()
  self.validate_publish_shape(tokens, pages, page_size, priority)
  if !self.publish_plan_matches(key, tokens, pages, priority) {
    raise PublishPlanInvalidated
  }
  let existing = self.find_request_root(key)
  let root = if existing >= 0 {
    existing
  } else {
    try! self.preflight_new_root(tokens.length(), pages.length())
    try! self.install_request_root(key)
  }
  let entry = try! self.insert_rooted(root, tokens, pages, page_size, priority)
  self.clear_publish_plan()
  entry
}

///|
/// Cancels an exact live publication plan before any radix mutation.
pub fn PrefixIndex::abort_publish(
  self : PrefixIndex,
  plan : PrefixPublishPlan,
) -> Unit raise PrefixIndexError {
  self.require_publish_plan(plan)
  self.clear_publish_plan()
}
