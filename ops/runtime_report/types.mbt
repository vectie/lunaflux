///|
/// Stable reasons an admitted runtime cannot be represented by the bounded
/// operator report. Errors contain no paths, descriptor data, or device
/// diagnostics.
pub(all) suberror RuntimeReportError {
  InvalidAdmission
  InconsistentAdmission
  DeviceBytesOverflow
} derive(Debug, Eq)

///|
/// Immutable scalar projection of one admitted runtime descriptor.
///
/// This value owns no filesystem root, device handle, allocation, mutable
/// runtime state, scheduler configuration, service configuration, or readiness
/// authority. Its fields remain private so reports can only be constructed
/// from admitted evidence by `build`.
pub struct RuntimeReport {
  priv weight_materialized_bytes : Int64
  priv weight_reserved_bytes : Int64
  priv weight_alignment_bytes : Int64
  priv activation_reserved_high_water_bytes : Int64
  priv workspace_offset_bytes : Int64
  priv workspace_bytes : Int64
  priv workspace_alignment_bytes : Int64
  priv activation_workspace_reserved_bytes : Int64
  priv kv_layout_version : Int
  priv kv_tokens_per_page : Int
  priv kv_page_count : Int
  priv kv_page_logical_bytes : Int64
  priv kv_page_reserved_stride_bytes : Int64
  priv kv_component_stride_bytes : Int64
  priv kv_layer_stride_bytes : Int64
  priv kv_segment_alignment_bytes : Int64
  priv kv_reserved_bytes : Int64
  priv kernel_module_file_bytes : Int64
  priv device_reserved_total_bytes : Int64
  priv graph_memory_availability : @runtime_resolved.RuntimeGraphMemoryAvailability
  priv graph_declared_upper_bound_bytes : Int64?
  priv graph_declared_alignment_bytes : Int64?
  priv graph_descriptor_ceiling_bytes : Int64?
  priv device_startup_upper_bound_bytes : Int64?
  priv max_prefill_rows : Int
  priv max_decode_rows : Int
  priv max_plan_rows : Int
  priv max_plan_tokens : Int
  priv max_plan_pages : Int
  priv max_completion_slots : Int
  priv max_capabilities : Int
  priv max_sequence_tokens : Int
  priv worker_max_token_id : Int
  priv max_input_tokens : Int
  priv max_new_tokens : Int
  priv max_context_tokens : Int
  priv inference_max_token_id : Int
}

///| Typed scalar projection of an admitted runtime instance foundation.

///|
/// It adds only policy-backed scheduler, service, tokenizer, preparation, and
/// telemetry capacities; it is not a readiness or OpenAI activation claim.
pub struct RuntimeInstanceReport {
  priv runtime : RuntimeReport
  priv tokenizer_vocabulary_size : Int
  priv scheduler_total_request_slots : Int
  priv scheduler_max_active_requests : Int
  priv scheduler_max_waiting_requests : Int
  priv scheduler_max_rows_per_step : Int
  priv scheduler_prefix_enabled : Bool
  priv service_max_request_bytes : Int
  priv preparation_lane_count : Int
  priv telemetry_capacity : Int
}

///|
/// Private scalar seam used to prove hostile and overflow behavior without
/// exposing a report-fabrication API.
priv struct RuntimeReportParts {
  weight_materialized_bytes : Int64
  weight_reserved_bytes : Int64
  weight_alignment_bytes : Int64
  activation_reserved_high_water_bytes : Int64
  workspace_offset_bytes : Int64
  workspace_bytes : Int64
  workspace_alignment_bytes : Int64
  activation_workspace_reserved_bytes : Int64
  kv_layout_version : Int
  kv_tokens_per_page : Int
  kv_page_count : Int
  kv_page_logical_bytes : Int64
  kv_page_reserved_stride_bytes : Int64
  kv_component_stride_bytes : Int64
  kv_layer_stride_bytes : Int64
  kv_segment_alignment_bytes : Int64
  kv_reserved_bytes : Int64
  kernel_module_file_bytes : Int64
  graph_memory_availability : @runtime_resolved.RuntimeGraphMemoryAvailability
  graph_declared_upper_bound_bytes : Int64?
  graph_declared_alignment_bytes : Int64?
  graph_descriptor_ceiling_bytes : Int64?
  graph_arena_reserved_total_bytes : Int64
  device_startup_upper_bound_bytes : Int64?
  max_prefill_rows : Int
  max_decode_rows : Int
  max_plan_rows : Int
  max_plan_tokens : Int
  max_plan_pages : Int
  max_completion_slots : Int
  max_capabilities : Int
  max_sequence_tokens : Int
  worker_max_token_id : Int
  max_input_tokens : Int
  max_new_tokens : Int
  max_context_tokens : Int
  inference_max_token_id : Int
  device_max_rows : Int
  device_max_tokens : Int
  device_max_pages : Int
  device_max_sequence_tokens : Int
  device_max_token_id : Int
}
