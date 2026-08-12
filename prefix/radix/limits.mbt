///|
fn valid_capacity(value : Int) -> Bool {
  value > 0 && value <= ABSOLUTE_MAX_PREFIX_CAPACITY
}

///|
pub fn PrefixIndexLimits::new(
  entry_capacity~ : Int,
  node_capacity~ : Int,
  max_tokens_per_entry~ : Int,
  page_capacity~ : Int,
  max_active_references_per_entry~ : Int,
  max_generation? : Int = ABSOLUTE_MAX_PREFIX_GENERATION,
) -> PrefixIndexLimits raise PrefixIndexError {
  if !valid_capacity(entry_capacity) {
    raise InvalidLimit(EntryCapacity)
  }
  if !valid_capacity(node_capacity) {
    raise InvalidLimit(NodeCapacity)
  }
  if !valid_capacity(max_tokens_per_entry) {
    raise InvalidLimit(TokensPerEntry)
  }
  if !valid_capacity(page_capacity) {
    raise InvalidLimit(PageCapacity)
  }
  if !valid_capacity(max_active_references_per_entry) {
    raise InvalidLimit(ActiveReferences)
  }
  if max_generation <= 0 || max_generation > ABSOLUTE_MAX_PREFIX_GENERATION {
    raise InvalidLimit(Generation)
  }
  {
    entry_capacity,
    node_capacity,
    max_tokens_per_entry,
    page_capacity,
    max_active_references_per_entry,
    max_generation,
  }
}

///|
pub fn PrefixIndexLimits::entry_capacity(self : PrefixIndexLimits) -> Int {
  self.entry_capacity
}

///|
pub fn PrefixIndexLimits::node_capacity(self : PrefixIndexLimits) -> Int {
  self.node_capacity
}

///|
pub fn PrefixIndexLimits::max_tokens_per_entry(self : PrefixIndexLimits) -> Int {
  self.max_tokens_per_entry
}

///|
pub fn PrefixIndexLimits::page_capacity(self : PrefixIndexLimits) -> Int {
  self.page_capacity
}

///|
pub fn PrefixIndexLimits::max_active_references_per_entry(
  self : PrefixIndexLimits,
) -> Int {
  self.max_active_references_per_entry
}

///|
pub fn PrefixIndexLimits::max_generation(self : PrefixIndexLimits) -> Int {
  self.max_generation
}
