///|
const ABSOLUTE_MAX_PREFIX_CAPACITY : Int32 = 1_048_576

///|
const ABSOLUTE_MAX_PREFIX_GENERATION : Int32 = 2_147_483_647

///|
#valtype
pub struct PrefixEntryId(Int32, Int32) derive(Debug, Eq, Hash)

///|
pub struct PrefixLayout {
  version : Int32
  tokens_per_page : Int32
} derive(Debug, Eq)

///|
pub struct PrefixIndexLimits {
  entry_capacity : Int32
  node_capacity : Int32
  token_capacity : Int32
  max_tokens_per_entry : Int32
  page_capacity : Int32
  max_scope_bytes : Int32
  max_active_references_per_entry : Int32
  max_generation : Int32
} derive(Debug, Eq)

///|
pub struct PrefixLookupBuffer {
  priv cells : FixedArray[Int32]
  priv hosts : FixedArray[@streaming_ir.HostPage]
  priv pages : @page_allocator.PageIdStorage
  priv mut page_count : Int32
  priv mut matched_tokens : Int32
  priv mut entry_index : Int32
  priv mut entry_generation : Int32
}

///|
pub struct PrefixEvictionBuffer {
  priv hosts : FixedArray[@streaming_ir.HostPage]
  priv pages : @page_allocator.PageIdStorage
  priv mut page_count : Int32
  priv mut entry_index : Int32
  priv mut entry_generation : Int32
}

///|
/// Reusable scalar token scratch owned by the enclosing scheduler.
pub struct PrefixTokenBuffer {
  priv tokens : FixedArray[Int32]
  priv mut length : Int32
}

///|
/// Reusable page publication scratch with per-page adoption evidence.
pub struct PrefixPublishBuffer {
  priv pages : @page_allocator.PageIdStorage
  priv planned : FixedArray[Bool]
  priv adopted : FixedArray[Bool]
  priv mut length : Int32
}

///|
/// Exact cache key retained by one tokenized request.
#valtype
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
  priv state : Int32
  priv owner : PrefixIndex
  priv generation : UInt64
  priv key : PrefixRequestKey
  priv entry_index : Int32
  priv entry_generation : Int32
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
  priv state : Int32
  priv owner : PrefixIndex
  priv generation : UInt64
}

///|
pub struct PrefixEntrySnapshot {
  id : PrefixEntryId
  token_count : Int32
  page_count : Int32
  active_references : Int32
  recency : UInt64
} derive(Debug, Eq)

///|
pub struct PrefixIndexSnapshot {
  entry_count : Int32
  retired_entry_count : Int32
  node_count : Int32
  page_count : Int32
  entries : ReadOnlyArray[PrefixEntrySnapshot]
} derive(Debug, Eq)

///|
pub struct PrefixIndex {
  priv page_hosts : FixedArray[@streaming_ir.HostPage]
  priv limits : PrefixIndexLimits
  priv root_used : FixedArray[Bool]
  priv root_models : FixedArray[@spec.ModelIdentity?]
  priv root_tokenizers : FixedArray[@tokenizer.TokenizerDigest?]
  priv root_layouts : FixedArray[PrefixLayout?]
  priv root_scope_lengths : FixedArray[Int32]
  priv root_scope_bytes : FixedArray[Byte]
  priv root_scope_scratch : FixedArray[Byte]
  priv root_entry_counts : FixedArray[Int32]
  priv root_child_heads : FixedArray[Int32]
  priv root_child_counts : FixedArray[Int32]
  priv root_free_next : FixedArray[Int32]
  priv mut root_free_head : Int32
  priv root_index_payloads : FixedArray[Int32]
  priv root_index_left : FixedArray[Int32]
  priv root_index_right : FixedArray[Int32]
  priv root_index_parents : FixedArray[Int32]
  priv root_index_heights : FixedArray[Int32]
  priv root_index_slots : FixedArray[Int32]
  priv root_index_free_next : FixedArray[Int32]
  priv mut root_index_root : Int32
  priv mut root_index_free_head : Int32
  priv node_used : FixedArray[Bool]
  priv node_parents : FixedArray[Int32]
  priv node_roots : FixedArray[Int32]
  priv node_usages : FixedArray[Int32]
  priv node_terminals : FixedArray[Int32]
  priv node_label_heads : FixedArray[Int32]
  priv node_label_tails : FixedArray[Int32]
  priv node_label_lengths : FixedArray[Int32]
  priv node_child_heads : FixedArray[Int32]
  priv node_child_counts : FixedArray[Int32]
  priv node_sibling_prev : FixedArray[Int32]
  priv node_sibling_next : FixedArray[Int32]
  priv node_protecting_entries : FixedArray[Int32]
  priv node_free_next : FixedArray[Int32]
  priv mut node_free_head : Int32
  priv child_index_payloads : FixedArray[Int32]
  priv child_index_left : FixedArray[Int32]
  priv child_index_right : FixedArray[Int32]
  priv child_index_parents : FixedArray[Int32]
  priv child_index_heights : FixedArray[Int32]
  priv child_index_slots : FixedArray[Int32]
  priv child_index_free_next : FixedArray[Int32]
  priv mut child_index_root : Int32
  priv mut child_index_free_head : Int32
  priv node_path_scratch : FixedArray[Int32]
  priv lookup_page_scratch : FixedArray[Int32]
  priv publish_token_scratch : FixedArray[Int32]
  priv publish_page_scratch : @page_allocator.PageIdStorage
  priv eviction_page_scratch : @page_allocator.PageIdStorage
  priv token_used : FixedArray[Bool]
  priv token_values : FixedArray[Int32]
  priv token_next : FixedArray[Int32]
  priv token_page_cells : FixedArray[Int32]
  priv token_free_next : FixedArray[Int32]
  priv mut token_free_head : Int32
  priv entry_states : FixedArray[Int32]
  priv entry_generations : FixedArray[Int32]
  priv entry_roots : FixedArray[Int32]
  priv entry_terminals : FixedArray[Int32]
  priv entry_token_counts : FixedArray[Int32]
  priv entry_page_counts : FixedArray[Int32]
  priv entry_priorities : FixedArray[Int32]
  priv entry_active_references : FixedArray[Int32]
  priv entry_recencies : FixedArray[UInt64]
  priv entry_free_next : FixedArray[Int32]
  priv mut entry_free_head : Int32
  priv eviction_heap : FixedArray[Int32]
  priv eviction_heap_positions : FixedArray[Int32]
  priv mut eviction_heap_count : Int32
  priv recency_entry_scratch : FixedArray[Int32]
  priv page_values : @page_allocator.PageIdStorage
  priv duplicate_page_scratch : @page_allocator.PageIdStorage
  priv page_free_next : FixedArray[Int32]
  priv mut page_free_head : Int32
  priv mut entry_count : Int32
  priv mut retired_entry_count : Int32
  priv mut node_count : Int32
  priv mut token_count : Int32
  priv mut page_count : Int32
  priv mut recency_clock : UInt64
  priv mut publish_plan_generation : UInt64
  priv mut publish_plan_active : Bool
  priv mut publish_plan_root : Int32
  priv mut publish_plan_token_count : Int32
  priv mut publish_plan_page_count : Int32
  priv mut publish_plan_priority : Int32
  priv mut eviction_plan_generation : UInt64
  priv mut eviction_plan_active : Bool
  priv mut eviction_plan_entry_index : Int32
  priv mut eviction_plan_entry_generation : Int32
  priv mut eviction_plan_page_count : Int32
  priv mut child_lookup_comparisons : Int32
  priv mut root_lookup_comparisons : Int32
  priv mut duplicate_page_comparisons : Int32
  priv mut duplicate_page_swaps : Int32
  priv mut recency_normalization_comparisons : Int32
  priv mut recency_normalization_swaps : Int32
  priv mut free_stack_touches : Int32
  priv mut eviction_heap_touches : Int32
  priv mut compaction_node_visits : Int32
  priv mut compaction_incident_edges : Int32
}
