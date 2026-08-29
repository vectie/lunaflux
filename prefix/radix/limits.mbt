///|
fn valid_capacity(value : Int) -> Bool {
  value > 0 && value <= ABSOLUTE_MAX_PREFIX_CAPACITY
}

///|
const ABSOLUTE_MAX_PREFIX_SCOPE_CELLS : Int = 16_777_216

///|
fn PrefixIndexLimits::validate(
  self : PrefixIndexLimits,
) -> Unit raise PrefixIndexError {
  if !valid_capacity(self.entry_capacity) {
    raise InvalidLimit(EntryCapacity)
  }
  if !valid_capacity(self.node_capacity) {
    raise InvalidLimit(NodeCapacity)
  }
  if !valid_capacity(self.token_capacity) {
    raise InvalidLimit(TokenCapacity)
  }
  if !valid_capacity(self.max_tokens_per_entry) {
    raise InvalidLimit(TokensPerEntry)
  }
  if self.entry_capacity > self.node_capacity {
    raise InvalidLimit(NodeCapacity)
  }
  if self.max_tokens_per_entry > self.token_capacity {
    raise InvalidLimit(TokenCapacity)
  }
  if !valid_capacity(self.page_capacity) {
    raise InvalidLimit(PageCapacity)
  }
  if !valid_capacity(self.max_scope_bytes) ||
    self.entry_capacity > ABSOLUTE_MAX_PREFIX_SCOPE_CELLS / self.max_scope_bytes {
    raise InvalidLimit(ScopeBytes)
  }
  if !valid_capacity(self.max_active_references_per_entry) {
    raise InvalidLimit(ActiveReferences)
  }
  if self.max_generation <= 0 ||
    self.max_generation > ABSOLUTE_MAX_PREFIX_GENERATION {
    raise InvalidLimit(Generation)
  }
}

///|
pub fn PrefixIndexLimits::new(
  entry_capacity~ : Int,
  node_capacity~ : Int,
  max_tokens_per_entry~ : Int,
  page_capacity~ : Int,
  max_active_references_per_entry~ : Int,
  token_capacity? : Int = node_capacity,
  max_scope_bytes? : Int = 256,
  max_generation? : Int = ABSOLUTE_MAX_PREFIX_GENERATION,
) -> PrefixIndexLimits raise PrefixIndexError {
  let limits = {
    entry_capacity,
    node_capacity,
    token_capacity,
    max_tokens_per_entry,
    page_capacity,
    max_scope_bytes,
    max_active_references_per_entry,
    max_generation,
  }
  limits.validate()
  limits
}

///|
pub fn PrefixIndexLimits::token_capacity(self : PrefixIndexLimits) -> Int {
  self.token_capacity
}

///|
pub fn PrefixIndexLimits::max_scope_bytes(self : PrefixIndexLimits) -> Int {
  self.max_scope_bytes
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

///|
fn prefix_checked_add(
  left : UInt64,
  right : UInt64,
) -> UInt64 raise PrefixIndexError {
  if left > 0xffffffffffffffffUL - right {
    raise InvalidLimit(TokenCapacity)
  }
  left + right
}

///|
fn prefix_checked_multiply(
  left : UInt64,
  right : UInt64,
) -> UInt64 raise PrefixIndexError {
  if right != 0UL && left > 0xffffffffffffffffUL / right {
    raise InvalidLimit(TokenCapacity)
  }
  left * right
}

///|
/// Exact scalar `Int` cells allocated by the compressed radix owner.
pub fn PrefixIndex::required_int_cells(
  limits : PrefixIndexLimits,
) -> UInt64 raise PrefixIndexError {
  limits.validate()
  let entries = limits.entry_capacity().to_uint64()
  let nodes = limits.node_capacity().to_uint64()
  let tokens = limits.token_capacity().to_uint64()
  let pages = limits.page_capacity().to_uint64()
  prefix_checked_add(
    prefix_checked_add(
      prefix_checked_multiply(entries, 25UL),
      prefix_checked_multiply(nodes, 21UL),
    ),
    prefix_checked_add(
      prefix_checked_multiply(tokens, 5UL),
      prefix_checked_multiply(pages, 10UL),
    ),
  )
}

///|
/// Exact boolean occupancy cells allocated by the compressed radix owner.
pub fn PrefixIndex::required_bool_cells(
  limits : PrefixIndexLimits,
) -> UInt64 raise PrefixIndexError {
  limits.validate()
  prefix_checked_add(
    prefix_checked_add(
      limits.entry_capacity().to_uint64(),
      limits.node_capacity().to_uint64(),
    ),
    prefix_checked_add(
      limits.token_capacity().to_uint64(),
      prefix_checked_multiply(limits.page_capacity().to_uint64(), 4UL),
    ),
  )
}

///|
/// Exact startup-copied security-scope byte cells.
pub fn PrefixIndex::required_byte_cells(
  limits : PrefixIndexLimits,
) -> UInt64 raise PrefixIndexError {
  limits.validate()
  prefix_checked_multiply(
    limits.entry_capacity().to_uint64() + 1UL,
    limits.max_scope_bytes().to_uint64(),
  )
}

///|
/// Exact identity reference cells retained by root metadata.
pub fn PrefixIndex::required_reference_cells(
  limits : PrefixIndexLimits,
) -> UInt64 raise PrefixIndexError {
  limits.validate()
  prefix_checked_multiply(limits.entry_capacity().to_uint64(), 3UL)
}
