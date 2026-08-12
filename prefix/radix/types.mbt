///|
const ABSOLUTE_MAX_PREFIX_CAPACITY : Int = 1_048_576

///|
const ABSOLUTE_MAX_PREFIX_GENERATION : Int = 2_147_483_647

///|
#valtype
pub struct PrefixEntryId(Int, Int) derive(Debug, Eq, Hash)

///|
pub struct PrefixLayout {
  version : Int
  tokens_per_page : Int
} derive(Debug, Eq)

///|
pub struct PrefixIdentity {
  model_identity : @spec.ModelIdentity
  tokenizer : @tokenizer.TokenizerDigest
  scope : @inference.CacheScope
  layout : PrefixLayout
}

///|
pub struct PrefixIndexLimits {
  entry_capacity : Int
  node_capacity : Int
  max_tokens_per_entry : Int
  page_capacity : Int
  max_active_references_per_entry : Int
  max_generation : Int
} derive(Debug, Eq)

///|
pub struct PrefixLookupBuffer {
  priv pages : FixedArray[@page_allocator.PageId?]
  priv mut page_count : Int
  priv mut matched_tokens : Int
  priv mut entry : PrefixEntryId?
}

///|
pub struct PrefixEvictionBuffer {
  priv pages : FixedArray[@page_allocator.PageId?]
  priv mut page_count : Int
  priv mut entry : PrefixEntryId?
}

///|
pub struct PrefixEntrySnapshot {
  id : PrefixEntryId
  token_count : Int
  page_count : Int
  active_references : Int
  recency : UInt64
} derive(Debug, Eq)

///|
pub struct PrefixIndexSnapshot {
  entry_count : Int
  retired_entry_count : Int
  node_count : Int
  page_count : Int
  entries : ReadOnlyArray[PrefixEntrySnapshot]
} derive(Debug, Eq)

///|
pub struct PrefixIndex {
  priv limits : PrefixIndexLimits
  priv roots : FixedArray[PrefixIdentity?]
  priv root_entry_counts : FixedArray[Int]
  priv node_used : FixedArray[Bool]
  priv node_tokens : FixedArray[Int]
  priv node_parents : FixedArray[Int]
  priv node_roots : FixedArray[Int]
  priv node_usages : FixedArray[Int]
  priv node_terminals : FixedArray[Int]
  priv entry_states : FixedArray[Int]
  priv entry_generations : FixedArray[Int]
  priv entry_roots : FixedArray[Int]
  priv entry_terminals : FixedArray[Int]
  priv entry_token_counts : FixedArray[Int]
  priv entry_page_heads : FixedArray[Int]
  priv entry_page_counts : FixedArray[Int]
  priv entry_active_references : FixedArray[Int]
  priv entry_recencies : FixedArray[UInt64]
  priv recency_scratch : FixedArray[UInt64]
  priv page_used : FixedArray[Bool]
  priv page_values : FixedArray[@page_allocator.PageId?]
  priv page_next : FixedArray[Int]
  priv mut entry_count : Int
  priv mut retired_entry_count : Int
  priv mut node_count : Int
  priv mut page_count : Int
  priv mut recency_clock : UInt64
}
