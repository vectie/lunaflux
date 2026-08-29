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
pub struct PrefixIndexLimits {
  entry_capacity : Int
  node_capacity : Int
  token_capacity : Int
  max_tokens_per_entry : Int
  page_capacity : Int
  max_scope_bytes : Int
  max_active_references_per_entry : Int
  max_generation : Int
} derive(Debug, Eq)

///|
pub struct PrefixLookupBuffer {
  priv pages : @page_allocator.PageIdStorage
  priv mut page_count : Int
  priv mut matched_tokens : Int
  priv mut entry_index : Int
  priv mut entry_generation : Int
}

///|
pub struct PrefixEvictionBuffer {
  priv pages : @page_allocator.PageIdStorage
  priv mut page_count : Int
  priv mut entry_index : Int
  priv mut entry_generation : Int
}

///|
/// Reusable scalar token scratch owned by the enclosing scheduler.
pub struct PrefixTokenBuffer {
  priv tokens : FixedArray[Int]
  priv mut length : Int
}

///|
/// Reusable page publication scratch with per-page adoption evidence.
pub struct PrefixPublishBuffer {
  priv pages : @page_allocator.PageIdStorage
  priv planned : FixedArray[Bool]
  priv adopted : FixedArray[Bool]
  priv mut length : Int
}

///|
/// Exact cache key retained by one tokenized request.
pub struct PrefixRequestKey {
  priv model_identity : @spec.ModelIdentity
  priv tokenizer : @tokenizer.TokenizerDigest
  priv layout : PrefixLayout
  priv semantic : @inference.LunaRequestPrefixView
}

///|
/// Exact one-shot authority for a preflighted publication transaction.
#valtype
pub struct PrefixPublishPlan {
  priv owner : PrefixIndex
  priv generation : UInt64
  priv key : PrefixRequestKey
}

///|
/// Allocation-free result of publication preflight on the warmed path.
#valtype
pub struct PrefixPublishStart {
  priv state : Int
  priv owner : PrefixIndex
  priv generation : UInt64
  priv key : PrefixRequestKey
  priv entry_index : Int
  priv entry_generation : Int
}

///|
/// Exact one-shot authority for a preflighted eviction transaction.
#valtype
pub struct PrefixEvictionPlan {
  priv owner : PrefixIndex
  priv generation : UInt64
}

///|
/// Allocation-free result of deterministic eviction selection.
#valtype
pub struct PrefixEvictionStart {
  priv state : Int
  priv owner : PrefixIndex
  priv generation : UInt64
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
  priv root_used : FixedArray[Bool]
  priv root_models : FixedArray[@spec.ModelIdentity?]
  priv root_tokenizers : FixedArray[@tokenizer.TokenizerDigest?]
  priv root_layouts : FixedArray[PrefixLayout?]
  priv root_scope_lengths : FixedArray[Int]
  priv root_scope_bytes : FixedArray[Byte]
  priv root_scope_scratch : FixedArray[Byte]
  priv root_entry_counts : FixedArray[Int]
  priv root_child_heads : FixedArray[Int]
  priv root_child_counts : FixedArray[Int]
  priv root_free_next : FixedArray[Int]
  priv mut root_free_head : Int
  priv root_index_payloads : FixedArray[Int]
  priv root_index_left : FixedArray[Int]
  priv root_index_right : FixedArray[Int]
  priv root_index_parents : FixedArray[Int]
  priv root_index_heights : FixedArray[Int]
  priv root_index_slots : FixedArray[Int]
  priv root_index_free_next : FixedArray[Int]
  priv mut root_index_root : Int
  priv mut root_index_free_head : Int
  priv node_used : FixedArray[Bool]
  priv node_parents : FixedArray[Int]
  priv node_roots : FixedArray[Int]
  priv node_usages : FixedArray[Int]
  priv node_terminals : FixedArray[Int]
  priv node_label_heads : FixedArray[Int]
  priv node_label_tails : FixedArray[Int]
  priv node_label_lengths : FixedArray[Int]
  priv node_child_heads : FixedArray[Int]
  priv node_child_counts : FixedArray[Int]
  priv node_sibling_prev : FixedArray[Int]
  priv node_sibling_next : FixedArray[Int]
  priv node_protecting_entries : FixedArray[Int]
  priv node_free_next : FixedArray[Int]
  priv mut node_free_head : Int
  priv child_index_payloads : FixedArray[Int]
  priv child_index_left : FixedArray[Int]
  priv child_index_right : FixedArray[Int]
  priv child_index_parents : FixedArray[Int]
  priv child_index_heights : FixedArray[Int]
  priv child_index_slots : FixedArray[Int]
  priv child_index_free_next : FixedArray[Int]
  priv mut child_index_root : Int
  priv mut child_index_free_head : Int
  priv node_path_scratch : FixedArray[Int]
  priv lookup_page_scratch : FixedArray[Int]
  priv publish_token_scratch : FixedArray[Int]
  priv publish_page_scratch : @page_allocator.PageIdStorage
  priv eviction_page_scratch : @page_allocator.PageIdStorage
  priv token_used : FixedArray[Bool]
  priv token_values : FixedArray[Int]
  priv token_next : FixedArray[Int]
  priv token_page_cells : FixedArray[Int]
  priv token_free_next : FixedArray[Int]
  priv mut token_free_head : Int
  priv entry_states : FixedArray[Int]
  priv entry_generations : FixedArray[Int]
  priv entry_roots : FixedArray[Int]
  priv entry_terminals : FixedArray[Int]
  priv entry_token_counts : FixedArray[Int]
  priv entry_page_counts : FixedArray[Int]
  priv entry_priorities : FixedArray[Int]
  priv entry_active_references : FixedArray[Int]
  priv entry_recencies : FixedArray[UInt64]
  priv entry_free_next : FixedArray[Int]
  priv mut entry_free_head : Int
  priv eviction_heap : FixedArray[Int]
  priv eviction_heap_positions : FixedArray[Int]
  priv mut eviction_heap_count : Int
  priv recency_entry_scratch : FixedArray[Int]
  priv page_values : @page_allocator.PageIdStorage
  priv duplicate_page_scratch : @page_allocator.PageIdStorage
  priv page_free_next : FixedArray[Int]
  priv mut page_free_head : Int
  priv mut entry_count : Int
  priv mut retired_entry_count : Int
  priv mut node_count : Int
  priv mut token_count : Int
  priv mut page_count : Int
  priv mut recency_clock : UInt64
  priv mut publish_plan_generation : UInt64
  priv mut publish_plan_active : Bool
  priv mut publish_plan_root : Int
  priv mut publish_plan_token_count : Int
  priv mut publish_plan_page_count : Int
  priv mut publish_plan_priority : Int
  priv mut eviction_plan_generation : UInt64
  priv mut eviction_plan_active : Bool
  priv mut eviction_plan_entry_index : Int
  priv mut eviction_plan_entry_generation : Int
  priv mut eviction_plan_page_count : Int
  priv mut child_lookup_comparisons : Int
  priv mut root_lookup_comparisons : Int
  priv mut duplicate_page_comparisons : Int
  priv mut duplicate_page_swaps : Int
  priv mut recency_normalization_comparisons : Int
  priv mut recency_normalization_swaps : Int
  priv mut free_stack_touches : Int
  priv mut eviction_heap_touches : Int
  priv mut compaction_node_visits : Int
  priv mut compaction_incident_edges : Int
}
