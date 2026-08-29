///|
/// Constructs a structurally valid entry identity for authenticated lookup.
pub fn PrefixEntryId::new(
  index~ : Int,
  generation~ : Int,
) -> PrefixEntryId raise PrefixIndexError {
  if index < 0 {
    raise InvalidLimit(EntryCapacity)
  }
  if generation <= 0 || generation > ABSOLUTE_MAX_PREFIX_GENERATION {
    raise InvalidLimit(Generation)
  }
  PrefixEntryId(index, generation)
}

///|
pub fn PrefixEntryId::index(self : PrefixEntryId) -> Int {
  self.0
}

///|
pub fn PrefixEntryId::generation(self : PrefixEntryId) -> Int {
  self.1
}

///|
pub fn PrefixLayout::new(
  version~ : Int,
  tokens_per_page~ : Int,
) -> PrefixLayout raise PrefixIndexError {
  let layout = { version, tokens_per_page }
  layout.validate()
  layout
}

///|
fn PrefixLayout::validate(self : PrefixLayout) -> Unit raise PrefixIndexError {
  if self.version <= 0 {
    raise InvalidLimit(LayoutVersion)
  }
  if self.tokens_per_page <= 0 ||
    self.tokens_per_page > ABSOLUTE_MAX_PREFIX_CAPACITY {
    raise InvalidLimit(TokensPerPage)
  }
}

///|
pub fn PrefixLayout::version(self : PrefixLayout) -> Int {
  self.version
}

///|
pub fn PrefixLayout::tokens_per_page(self : PrefixLayout) -> Int {
  self.tokens_per_page
}

///|
pub fn PrefixRequestKey::new(
  model_identity~ : @spec.ModelIdentity,
  tokenizer~ : @tokenizer.TokenizerDigest,
  layout~ : PrefixLayout,
  semantic~ : @inference.LunaRequestPrefixView,
) -> PrefixRequestKey raise PrefixIndexError {
  layout.validate()
  if !semantic.is_live() {
    raise StaleKey
  }
  { model_identity, tokenizer, layout, semantic }
}

///|
pub fn PrefixRequestKey::is_live(self : PrefixRequestKey) -> Bool {
  self.semantic.is_live()
}

///|
pub fn PrefixRequestKey::cache_permission(
  self : PrefixRequestKey,
) -> @inference.CachePermission raise PrefixIndexError {
  self.semantic.cache_permission() catch {
    _ => raise StaleKey
  }
}
