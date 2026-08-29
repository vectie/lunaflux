///|
/// Single thread-confined owner of request, physical-page, and block-table
/// metadata. Every mutable collection is fixed at startup.
pub struct Scheduler {
  priv capacity : @runtime_resolved.ResolvedRuntimeCapacity
  priv worker_limits : @worker_protocol.WorkerProtocolLimits
  priv model_identity : @spec.ModelIdentity
  priv page_allocator : @page_allocator.PageAllocator
  priv block_tables : @block_table.BlockTableArena
  priv prefix_index : @radix.PrefixIndex?
  priv prefix_tokens : @radix.PrefixTokenBuffer?
  priv prefix_pages : @radix.PrefixPublishBuffer?
  priv prefix_lookup : @radix.PrefixLookupBuffer?
  priv prefix_eviction : @radix.PrefixEvictionBuffer?
  priv model_plan_generation : @worker_protocol.ModelPlanGeneration
  priv intermediate_prefill_recipe : @worker_protocol.RowCapabilityRecipe
  priv final_prefill_recipe : @worker_protocol.RowCapabilityRecipe
  priv decode_recipe : @worker_protocol.RowCapabilityRecipe
  priv completion_slots : FixedArray[@worker_protocol.CompletionSlot]
  priv plan_a : @worker_protocol.SchedulePlanBuffer
  priv plan_b : @worker_protocol.SchedulePlanBuffer
  priv plan_buffer_binding : SchedulerPlanBufferBinding
  priv completion_a : @worker_protocol.CompletionBuffer
  priv completion_b : @worker_protocol.CompletionBuffer
  priv request_tables : @block_table.BlockTableIdStorage
  priv journal_tables : @block_table.BlockTableIdStorage
  priv page_run : @page_allocator.PageRunBuffer
  priv release_pages : @block_table.BlockTablePageBuffer
  priv slot_states : FixedArray[Int]
  priv slot_generations : FixedArray[Int]
  priv requests : FixedArray[TokenizedRequest?]
  priv request_stop_tokens : FixedArray[
    @inference.LunaRequestStopTokenRetentionSlot,
  ]
  priv request_generations : FixedArray[@inference.RequestGeneration]
  priv deadlines : FixedArray[UInt64]
  priv admitted_steps : FixedArray[UInt64]
  priv processed_inputs : FixedArray[Int]
  priv prefill_cursors : FixedArray[Int]
  priv output_counts : FixedArray[Int]
  priv in_flight : FixedArray[Bool]
  priv decode_tokens : FixedArray[Int]
  priv generated_history : FixedArray[Int]
  priv recompute_required : FixedArray[Bool]
  priv recompute_processed : FixedArray[Int]
  priv prefix_entry_indices : FixedArray[Int]
  priv prefix_entry_generations : FixedArray[Int]
  priv prefix_matched_tokens : FixedArray[Int]
  priv prefix_candidate_entry_indices : FixedArray[Int]
  priv prefix_candidate_entry_generations : FixedArray[Int]
  priv prefix_candidate_tokens : FixedArray[Int]
  priv prefix_selection_pages : @page_allocator.PageIdStorage
  priv prefix_selection_offsets : FixedArray[Int]
  priv prefix_selection_counts : FixedArray[Int]
  priv prefix_selection_entries : FixedArray[Int]
  priv prefix_selection_entry_generations : FixedArray[Int]
  priv waiting_prev : FixedArray[Int]
  priv waiting_next : FixedArray[Int]
  priv waiting_link_generations : FixedArray[Int]
  priv waiting_selection_heap : FixedArray[Int]
  priv waiting_selection_ranks : FixedArray[Int]
  priv deadline_heap_slots : FixedArray[Int]
  priv deadline_heap_generations : FixedArray[Int]
  priv deadline_heap_positions : FixedArray[Int]
  priv deadline_due_heap : FixedArray[Int]
  priv free_next : FixedArray[Int]
  priv active_slots : FixedArray[Int]
  priv active_positions : FixedArray[Int]
  priv prefill_slots : FixedArray[Int]
  priv prefill_tokens : FixedArray[Int]
  priv decode_slots : FixedArray[Int]
  priv journal_slots : FixedArray[Int]
  priv journal_old_page_counts : FixedArray[Int]
  priv journal_appended : FixedArray[Bool]
  priv plan_a_slots : FixedArray[Int]
  priv plan_b_slots : FixedArray[Int]
  priv terminal_request_ids : FixedArray[@inference.RequestId]
  priv terminal_generations : FixedArray[@inference.RequestGeneration]
  priv terminal_reasons : FixedArray[TerminalReason]
  priv terminal_cached_input_tokens : FixedArray[Int]
  priv terminal_sequences : FixedArray[UInt64]
  priv terminal_slot_indices : FixedArray[Int]
  priv terminal_slot_generations : FixedArray[Int]
  priv terminal_route_lanes : FixedArray[Int]
  priv terminal_route_generations : FixedArray[UInt64]
  priv output_request_ids : FixedArray[@inference.RequestId]
  priv output_generations : FixedArray[@inference.RequestGeneration]
  priv output_token_ids : FixedArray[Int]
  priv output_positions : FixedArray[Int]
  priv output_sequences : FixedArray[UInt64]
  priv output_slot_indices : FixedArray[Int]
  priv output_slot_generations : FixedArray[Int]
  priv output_route_lanes : FixedArray[Int]
  priv output_route_generations : FixedArray[UInt64]
  priv publication_route_lanes : FixedArray[Int]
  priv publication_route_generations : FixedArray[UInt64]
  priv mut free_head : Int
  priv mut free_tail : Int
  priv mut waiting_head : Int
  priv mut waiting_tail : Int
  priv mut waiting_count : Int
  priv mut active_count : Int
  priv mut deadline_heap_count : Int
  priv mut live_count : Int
  priv mut terminal_head : Int
  priv mut terminal_tail : Int
  priv mut terminal_count : Int
  priv mut output_head : Int
  priv mut output_tail : Int
  priv mut output_count : Int
  priv mut publication_cursor : UInt64
  priv mut generation_cursor : UInt64
  priv mut step : UInt64
  priv mut last_plan_sequence : UInt64
  priv mut last_retired_plan_sequence : UInt64
  priv mut telemetry_plan_sequence : UInt64
  priv mut telemetry_plan_rows : Int
  priv mut telemetry_plan_tokens : Int
  priv mut plan_a_sequence : UInt64
  priv mut plan_b_sequence : UInt64
  priv mut prefill_count : Int
  priv mut decode_count : Int
  priv mut journal_count : Int
  priv mut preemption_cursor : Int
  priv mut decode_cursor : Int
  priv mut pending_decode_cursor : Int
  priv mut waiting_selection_count : Int
  priv mut waiting_unlink_link_touches : Int
  priv mut selection_waiting_visits : Int
  priv mut selection_active_visits : Int
  priv mut prefix_selection_page_count : Int
  priv mut deadline_live_visits : Int
  priv mut deadline_due_pops : Int
  priv mut cancel_reserved : Bool
  priv mut exclusive_admission_reserved : Bool
  priv mut cancel_slot : Int
  priv mut cancel_slot_generation : Int
  priv mut cancel_request_id : @inference.RequestId
  priv mut cancel_request_generation : @inference.RequestGeneration
  priv mut cancel_publication_cursor : UInt64
  priv mut cancel_step : UInt64
  priv mut cancel_token_taken : Bool
  priv mut prefix_lookups : UInt64
  priv mut prefix_hits : UInt64
  priv mut prefix_misses : UInt64
  priv mut prefix_evictions : UInt64
  priv mut prefix_tokens_reused : UInt64
  priv mut prefix_tokens_computed : UInt64
  priv mut prefix_publications : UInt64
  priv mut prefix_candidate_discoveries : UInt64
}
