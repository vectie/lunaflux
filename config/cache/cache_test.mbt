///|
fn cache(
  tokens_per_page? : Int = 16,
  total_page_count? : Int = 4096,
  block_table_pages_per_request? : Int = 1024,
  prefix_enabled? : Bool = true,
  max_prefix_entries? : Int = 1024,
  max_prefix_nodes? : Int = 4096,
  max_prefix_tokens_per_entry? : Int = 4096,
  max_prefix_pages? : Int = 1024,
  max_active_references_per_page? : Int = 1024,
  max_cached_references_per_page? : Int = 1024,
  layout_version? : Int = 1,
) -> @cache.CacheConfig raise @cache.CacheConfigError {
  @cache.new(
    tokens_per_page~,
    total_page_count~,
    block_table_pages_per_request~,
    prefix_enabled~,
    max_prefix_entries~,
    max_prefix_nodes~,
    max_prefix_tokens_per_entry~,
    max_prefix_pages~,
    max_active_references_per_page~,
    max_cached_references_per_page~,
    layout_version~,
  )
}

///|
fn expect_cache_error(
  run : () -> @cache.CacheConfig raise @cache.CacheConfigError,
  expected : @cache.CacheConfigError,
) -> Unit raise {
  try run() catch {
    error => assert_eq(error, expected)
  } noraise {
    _ => fail("expected cache configuration rejection")
  }
}

///|
test "cache preserves exact values and inclusive minima" {
  let config = cache(
    tokens_per_page=1,
    total_page_count=1,
    block_table_pages_per_request=1,
    max_prefix_entries=1,
    max_prefix_nodes=1,
    max_prefix_tokens_per_entry=1,
    max_prefix_pages=1,
    max_active_references_per_page=1,
    max_cached_references_per_page=1,
    layout_version=1,
  )
  assert_eq(config.tokens_per_page(), 1)
  assert_eq(config.total_page_count(), 1)
  assert_eq(config.block_table_pages_per_request(), 1)
  assert_true(config.prefix_enabled())
  assert_eq(config.max_prefix_entries(), 1)
  assert_eq(config.max_prefix_nodes(), 1)
  assert_eq(config.max_prefix_tokens_per_entry(), 1)
  assert_eq(config.max_prefix_pages(), 1)
  assert_eq(config.max_active_references_per_page(), 1)
  assert_eq(config.max_cached_references_per_page(), 1)
  assert_eq(config.layout_version(), 1)
}

///|
test "cache accepts every inclusive hard ceiling" {
  let config = cache(
    tokens_per_page=262_144,
    total_page_count=1_048_576,
    block_table_pages_per_request=262_144,
    max_prefix_entries=1_048_576,
    max_prefix_nodes=1_048_576,
    max_prefix_tokens_per_entry=1_048_576,
    max_prefix_pages=1_048_576,
    max_active_references_per_page=1_048_576,
    max_cached_references_per_page=1_048_576,
    layout_version=1_000_000,
  )
  assert_eq(config.tokens_per_page(), 262_144)
  assert_eq(config.total_page_count(), 1_048_576)
  assert_eq(config.block_table_pages_per_request(), 262_144)
  assert_eq(config.max_prefix_entries(), 1_048_576)
  assert_eq(config.max_prefix_nodes(), 1_048_576)
  assert_eq(config.max_prefix_tokens_per_entry(), 1_048_576)
  assert_eq(config.max_prefix_pages(), 1_048_576)
  assert_eq(config.max_active_references_per_page(), 1_048_576)
  assert_eq(config.max_cached_references_per_page(), 1_048_576)
  assert_eq(config.layout_version(), 1_000_000)
}

///|
test "cache accepts an explicitly disabled zero-capacity prefix index" {
  let config = cache(
    prefix_enabled=false,
    max_prefix_entries=0,
    max_prefix_nodes=0,
    max_prefix_tokens_per_entry=0,
    max_prefix_pages=0,
  )
  assert_false(config.prefix_enabled())
  assert_eq(config.max_prefix_entries(), 0)
  assert_eq(config.max_prefix_nodes(), 0)
  assert_eq(config.max_prefix_tokens_per_entry(), 0)
  assert_eq(config.max_prefix_pages(), 0)
}

///|
test "cache keeps logical prefix page cells independent of physical pages" {
  let config = cache(
    tokens_per_page=16,
    total_page_count=1,
    block_table_pages_per_request=1,
    max_prefix_entries=1,
    max_prefix_nodes=32,
    max_prefix_tokens_per_entry=32,
    max_prefix_pages=1024,
  )
  assert_eq(config.total_page_count(), 1)
  assert_eq(config.max_prefix_pages(), 1024)
  assert_eq(config.max_prefix_tokens_per_entry(), 32)
}

///|
test "cache rejects every ordinary numeric lower bound" {
  expect_cache_error(
    () => cache(tokens_per_page=0),
    Invalid(field=TokensPerPage, issue=BelowMinimum),
  )
  expect_cache_error(
    () => cache(total_page_count=0),
    Invalid(field=TotalPageCount, issue=BelowMinimum),
  )
  expect_cache_error(
    () => cache(block_table_pages_per_request=0),
    Invalid(field=BlockTablePagesPerRequest, issue=BelowMinimum),
  )
  expect_cache_error(
    () => cache(max_active_references_per_page=0),
    Invalid(field=MaxActiveReferencesPerPage, issue=BelowMinimum),
  )
  expect_cache_error(
    () => cache(max_cached_references_per_page=0),
    Invalid(field=MaxCachedReferencesPerPage, issue=BelowMinimum),
  )
  expect_cache_error(
    () => cache(layout_version=0),
    Invalid(field=LayoutVersion, issue=BelowMinimum),
  )
}

///|
test "cache requires every enabled prefix capacity" {
  expect_cache_error(
    () => cache(max_prefix_entries=0),
    Invalid(field=MaxPrefixEntries, issue=RequiredWhenPrefixEnabled),
  )
  expect_cache_error(
    () => cache(max_prefix_nodes=0),
    Invalid(field=MaxPrefixNodes, issue=RequiredWhenPrefixEnabled),
  )
  expect_cache_error(
    () => cache(max_prefix_tokens_per_entry=0),
    Invalid(field=MaxPrefixTokensPerEntry, issue=RequiredWhenPrefixEnabled),
  )
  expect_cache_error(
    () => cache(max_prefix_pages=0),
    Invalid(field=MaxPrefixPages, issue=RequiredWhenPrefixEnabled),
  )
}

///|
test "cache rejects every hard-ceiling breach" {
  expect_cache_error(
    () => cache(tokens_per_page=262_145),
    Invalid(field=TokensPerPage, issue=AboveMaximum),
  )
  expect_cache_error(
    () => cache(total_page_count=1_048_577),
    Invalid(field=TotalPageCount, issue=AboveMaximum),
  )
  expect_cache_error(
    () => cache(block_table_pages_per_request=262_145),
    Invalid(field=BlockTablePagesPerRequest, issue=AboveMaximum),
  )
  expect_cache_error(
    () => cache(max_prefix_entries=1_048_577),
    Invalid(field=MaxPrefixEntries, issue=AboveMaximum),
  )
  expect_cache_error(
    () => cache(max_prefix_nodes=1_048_577),
    Invalid(field=MaxPrefixNodes, issue=AboveMaximum),
  )
  expect_cache_error(
    () => cache(max_prefix_tokens_per_entry=1_048_577),
    Invalid(field=MaxPrefixTokensPerEntry, issue=AboveMaximum),
  )
  expect_cache_error(
    () => cache(max_prefix_pages=1_048_577),
    Invalid(field=MaxPrefixPages, issue=AboveMaximum),
  )
  expect_cache_error(
    () => cache(max_active_references_per_page=1_048_577),
    Invalid(field=MaxActiveReferencesPerPage, issue=AboveMaximum),
  )
  expect_cache_error(
    () => cache(max_cached_references_per_page=1_048_577),
    Invalid(field=MaxCachedReferencesPerPage, issue=AboveMaximum),
  )
  expect_cache_error(
    () => cache(layout_version=1_000_001),
    Invalid(field=LayoutVersion, issue=AboveMaximum),
  )
}

///|
test "cache rejects every internal capacity relation breach" {
  expect_cache_error(
    () => cache(total_page_count=1023, block_table_pages_per_request=1024),
    Invalid(field=BlockTablePagesPerRequest, issue=ExceedsTotalPageCount),
  )
  expect_cache_error(
    () => cache(max_prefix_entries=4097, max_prefix_nodes=4096),
    Invalid(field=MaxPrefixEntries, issue=ExceedsPrefixNodeCapacity),
  )
  expect_cache_error(
    () => {
      cache(
        max_prefix_entries=1,
        max_prefix_nodes=4096,
        max_prefix_tokens_per_entry=4097,
      )
    },
    Invalid(field=MaxPrefixTokensPerEntry, issue=ExceedsPrefixNodeCapacity),
  )
}

///|
test "cache rejects any disabled prefix capacity" {
  expect_cache_error(
    () => {
      cache(
        prefix_enabled=false,
        max_prefix_entries=1,
        max_prefix_nodes=0,
        max_prefix_tokens_per_entry=0,
        max_prefix_pages=0,
      )
    },
    Invalid(field=MaxPrefixEntries, issue=MustBeZeroWhenPrefixDisabled),
  )
  expect_cache_error(
    () => {
      cache(
        prefix_enabled=false,
        max_prefix_entries=0,
        max_prefix_nodes=1,
        max_prefix_tokens_per_entry=0,
        max_prefix_pages=0,
      )
    },
    Invalid(field=MaxPrefixNodes, issue=MustBeZeroWhenPrefixDisabled),
  )
  expect_cache_error(
    () => {
      cache(
        prefix_enabled=false,
        max_prefix_entries=0,
        max_prefix_nodes=0,
        max_prefix_tokens_per_entry=1,
        max_prefix_pages=0,
      )
    },
    Invalid(field=MaxPrefixTokensPerEntry, issue=MustBeZeroWhenPrefixDisabled),
  )
  expect_cache_error(
    () => {
      cache(
        prefix_enabled=false,
        max_prefix_entries=0,
        max_prefix_nodes=0,
        max_prefix_tokens_per_entry=0,
        max_prefix_pages=1,
      )
    },
    Invalid(field=MaxPrefixPages, issue=MustBeZeroWhenPrefixDisabled),
  )
}
