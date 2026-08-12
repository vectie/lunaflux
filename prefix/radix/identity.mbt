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
  if version <= 0 {
    raise InvalidLimit(LayoutVersion)
  }
  if tokens_per_page <= 0 || tokens_per_page > ABSOLUTE_MAX_PREFIX_CAPACITY {
    raise InvalidLimit(TokensPerPage)
  }
  { version, tokens_per_page }
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
pub fn PrefixIdentity::new(
  model_identity~ : @spec.ModelIdentity,
  tokenizer~ : @tokenizer.TokenizerDigest,
  scope~ : @inference.CacheScope,
  layout~ : PrefixLayout,
) -> PrefixIdentity {
  { model_identity, tokenizer, scope, layout }
}

///|
pub fn PrefixIdentity::model_identity(
  self : PrefixIdentity,
) -> @spec.ModelIdentity {
  self.model_identity
}

///|
pub fn PrefixIdentity::tokenizer(
  self : PrefixIdentity,
) -> @tokenizer.TokenizerDigest {
  self.tokenizer
}

///|
pub fn PrefixIdentity::scope(self : PrefixIdentity) -> @inference.CacheScope {
  self.scope
}

///|
pub fn PrefixIdentity::layout(self : PrefixIdentity) -> PrefixLayout {
  self.layout
}
