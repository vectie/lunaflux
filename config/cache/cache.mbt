///|
const MAX_TOKENS_PER_PAGE : Int32 = 262_144

///|
const MAX_TOTAL_PAGE_COUNT : Int32 = 1_048_576

///|
const MAX_BLOCK_TABLE_PAGES_PER_REQUEST : Int32 = 262_144

///|
const MAX_PREFIX_CAPACITY : Int32 = 1_048_576

///|
const MAX_REFERENCES_PER_PAGE : Int32 = 1_048_576

///|
const MAX_LAYOUT_VERSION : Int32 = 1_000_000

///|
/// Fields accepted by the cache configuration constructor.
pub(all) enum CacheField {
  TokensPerPage
  TotalPageCount
  BlockTablePagesPerRequest
  MaxPrefixEntries
  MaxPrefixNodes
  MaxPrefixTokensPerEntry
  MaxPrefixPages
  MaxPrefixScopeBytes
  MaxActiveReferencesPerPage
  MaxCachedReferencesPerPage
  LayoutVersion
} derive(Debug, Eq)

///|
/// Bounded validation categories for cache settings.
pub(all) enum CacheIssue {
  BelowMinimum
  AboveMaximum
  ExceedsTotalPageCount
  RequiredWhenPrefixEnabled
  MustBeZeroWhenPrefixDisabled
  ExceedsPrefixNodeCapacity
} derive(Debug, Eq)

///|
/// Payload-safe cache configuration validation failure.
pub(all) suberror CacheConfigError {
  Invalid(field~ : CacheField, issue~ : CacheIssue)
} derive(Debug, Eq)

///|
/// Immutable physical page and optional prefix-index capacity settings.
pub struct CacheConfig {
  tokens_per_page : Int32
  total_page_count : Int32
  block_table_pages_per_request : Int32
  prefix_enabled : Bool
  max_prefix_entries : Int32
  max_prefix_nodes : Int32
  max_prefix_tokens_per_entry : Int32
  max_prefix_pages : Int32
  max_prefix_scope_bytes : Int32
  max_active_references_per_page : Int32
  max_cached_references_per_page : Int32
  layout_version : Int32
} derive(Debug, Eq)

///|
/// Constructs bounded cache settings.
///
/// Disabled prefix caching owns no prefix metadata capacity. When enabled,
/// entry, aggregate node, per-entry token, and logical page-cell capacities
/// are all explicit and nonzero.
pub fn new(
  tokens_per_page~ : Int32,
  total_page_count~ : Int32,
  block_table_pages_per_request~ : Int32,
  prefix_enabled~ : Bool,
  max_prefix_entries~ : Int32,
  max_prefix_nodes~ : Int32,
  max_prefix_tokens_per_entry~ : Int32,
  max_prefix_pages~ : Int32,
  max_active_references_per_page~ : Int32,
  max_cached_references_per_page~ : Int32,
  layout_version~ : Int32,
  max_prefix_scope_bytes~ : Int32 = 0,
) -> CacheConfig raise CacheConfigError {
  require_positive_bounded(tokens_per_page, MAX_TOKENS_PER_PAGE, TokensPerPage)
  require_positive_bounded(
    total_page_count,
    MAX_TOTAL_PAGE_COUNT,
    TotalPageCount,
  )
  require_positive_bounded(
    block_table_pages_per_request,
    MAX_BLOCK_TABLE_PAGES_PER_REQUEST,
    BlockTablePagesPerRequest,
  )
  require_positive_bounded(
    max_active_references_per_page,
    MAX_REFERENCES_PER_PAGE,
    MaxActiveReferencesPerPage,
  )
  require_positive_bounded(
    max_cached_references_per_page,
    MAX_REFERENCES_PER_PAGE,
    MaxCachedReferencesPerPage,
  )
  require_positive_bounded(layout_version, MAX_LAYOUT_VERSION, LayoutVersion)
  require_prefix_capacity(max_prefix_entries, MaxPrefixEntries, prefix_enabled)
  require_prefix_capacity(max_prefix_nodes, MaxPrefixNodes, prefix_enabled)
  require_prefix_capacity(
    max_prefix_tokens_per_entry,
    MaxPrefixTokensPerEntry,
    prefix_enabled,
  )
  require_prefix_capacity(max_prefix_pages, MaxPrefixPages, prefix_enabled)
  require_prefix_capacity(
    max_prefix_scope_bytes,
    MaxPrefixScopeBytes,
    prefix_enabled,
  )
  if block_table_pages_per_request > total_page_count {
    raise Invalid(field=BlockTablePagesPerRequest, issue=ExceedsTotalPageCount)
  }
  if prefix_enabled {
    if max_prefix_entries > max_prefix_nodes {
      raise Invalid(field=MaxPrefixEntries, issue=ExceedsPrefixNodeCapacity)
    }
    if max_prefix_tokens_per_entry > max_prefix_nodes {
      raise Invalid(
        field=MaxPrefixTokensPerEntry,
        issue=ExceedsPrefixNodeCapacity,
      )
    }
  }
  {
    tokens_per_page,
    total_page_count,
    block_table_pages_per_request,
    prefix_enabled,
    max_prefix_entries,
    max_prefix_nodes,
    max_prefix_tokens_per_entry,
    max_prefix_pages,
    max_prefix_scope_bytes,
    max_active_references_per_page,
    max_cached_references_per_page,
    layout_version,
  }
}

///|
pub fn CacheConfig::max_prefix_scope_bytes(self : CacheConfig) -> Int32 {
  self.max_prefix_scope_bytes
}

///|
fn require_positive_bounded(
  value : Int32,
  maximum : Int32,
  field : CacheField,
) -> Unit raise CacheConfigError {
  if value <= 0 {
    raise Invalid(field~, issue=BelowMinimum)
  }
  if value > maximum {
    raise Invalid(field~, issue=AboveMaximum)
  }
}

///|
fn require_prefix_capacity(
  value : Int32,
  field : CacheField,
  prefix_enabled : Bool,
) -> Unit raise CacheConfigError {
  if prefix_enabled {
    if value <= 0 {
      raise Invalid(field~, issue=RequiredWhenPrefixEnabled)
    }
    if value > MAX_PREFIX_CAPACITY {
      raise Invalid(field~, issue=AboveMaximum)
    }
  } else if value != 0 {
    raise Invalid(field~, issue=MustBeZeroWhenPrefixDisabled)
  }
}

///|
pub fn CacheConfig::tokens_per_page(self : CacheConfig) -> Int32 {
  self.tokens_per_page
}

///|
pub fn CacheConfig::total_page_count(self : CacheConfig) -> Int32 {
  self.total_page_count
}

///|
pub fn CacheConfig::block_table_pages_per_request(self : CacheConfig) -> Int32 {
  self.block_table_pages_per_request
}

///|
pub fn CacheConfig::prefix_enabled(self : CacheConfig) -> Bool {
  self.prefix_enabled
}

///|
pub fn CacheConfig::max_prefix_entries(self : CacheConfig) -> Int32 {
  self.max_prefix_entries
}

///|
pub fn CacheConfig::max_prefix_nodes(self : CacheConfig) -> Int32 {
  self.max_prefix_nodes
}

///|
pub fn CacheConfig::max_prefix_tokens_per_entry(self : CacheConfig) -> Int32 {
  self.max_prefix_tokens_per_entry
}

///|
pub fn CacheConfig::max_prefix_pages(self : CacheConfig) -> Int32 {
  self.max_prefix_pages
}

///|
pub fn CacheConfig::max_active_references_per_page(self : CacheConfig) -> Int32 {
  self.max_active_references_per_page
}

///|
pub fn CacheConfig::max_cached_references_per_page(self : CacheConfig) -> Int32 {
  self.max_cached_references_per_page
}

///|
pub fn CacheConfig::layout_version(self : CacheConfig) -> Int32 {
  self.layout_version
}

///|
pub extend CacheField with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend CacheField with Eq::{not_equal, equal}

///|
pub extend CacheIssue with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend CacheIssue with Eq::{not_equal, equal}

///|
pub extend CacheConfigError with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend CacheConfigError with Eq::{not_equal, equal}

///|
pub extend CacheConfig with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend CacheConfig with Eq::{not_equal, equal}
