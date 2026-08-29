///|
const MAX_INT64 : Int64 = 0x7fffffffffffffffL

///|
fn checked_nonnegative_add(
  left : Int64,
  right : Int64,
) -> Int64 raise RuntimeReportError {
  if left < 0L || right < 0L {
    raise InvalidAdmission
  }
  if left > MAX_INT64 - right {
    raise DeviceBytesOverflow
  }
  left + right
}

///|
fn validate_parts(parts : RuntimeReportParts) -> Unit raise RuntimeReportError {
  if parts.weight_materialized_bytes < 0L ||
    parts.weight_reserved_bytes < parts.weight_materialized_bytes ||
    parts.weight_alignment_bytes <= 0L ||
    parts.activation_reserved_high_water_bytes < 0L ||
    parts.activation_workspace_reserved_bytes <
    parts.activation_reserved_high_water_bytes ||
    parts.workspace_offset_bytes < 0L ||
    parts.workspace_bytes < 0L ||
    parts.workspace_alignment_bytes <= 0L ||
    parts.kv_layout_version <= 0 ||
    parts.kv_tokens_per_page <= 0 ||
    parts.kv_page_count <= 0 ||
    parts.kv_page_logical_bytes < 0L ||
    parts.kv_page_reserved_stride_bytes < parts.kv_page_logical_bytes ||
    parts.kv_component_stride_bytes < parts.kv_page_reserved_stride_bytes ||
    parts.kv_layer_stride_bytes < parts.kv_component_stride_bytes ||
    parts.kv_segment_alignment_bytes <= 0L ||
    parts.kv_reserved_bytes < parts.kv_layer_stride_bytes ||
    parts.kernel_module_file_bytes < 0L ||
    parts.max_prefill_rows <= 0 ||
    parts.max_decode_rows <= 0 ||
    parts.max_plan_rows <= 0 ||
    parts.max_plan_tokens <= 0 ||
    parts.max_plan_pages <= 0 ||
    parts.max_completion_slots < parts.max_plan_rows ||
    parts.max_capabilities <= 0 ||
    parts.max_sequence_tokens <= 0 ||
    parts.worker_max_token_id < 0 ||
    parts.max_input_tokens <= 0 ||
    parts.max_new_tokens <= 0 ||
    parts.max_context_tokens <= 0 ||
    parts.max_input_tokens > parts.max_context_tokens ||
    parts.max_new_tokens > parts.max_context_tokens ||
    parts.max_context_tokens > parts.max_sequence_tokens ||
    parts.inference_max_token_id < 0 ||
    parts.inference_max_token_id > parts.worker_max_token_id {
    raise InvalidAdmission
  }
  let workspace_end = checked_nonnegative_add(
    parts.workspace_offset_bytes,
    parts.workspace_bytes,
  )
  if workspace_end > parts.activation_workspace_reserved_bytes {
    raise InvalidAdmission
  }
  if parts.device_max_rows != parts.max_plan_rows ||
    parts.device_max_tokens != parts.max_plan_tokens ||
    parts.device_max_pages != parts.max_plan_pages ||
    parts.device_max_sequence_tokens != parts.max_sequence_tokens ||
    parts.device_max_token_id != parts.worker_max_token_id {
    raise InconsistentAdmission
  }
  match parts.graph_memory_availability {
    GraphAccountingAbsent | GraphAccountingUnsupported =>
      if parts.graph_declared_upper_bound_bytes is Some(_) ||
        parts.graph_declared_alignment_bytes is Some(_) ||
        parts.graph_descriptor_ceiling_bytes is Some(_) ||
        parts.device_startup_upper_bound_bytes is Some(_) {
        raise InconsistentAdmission
      }
    GraphAccountingBounded => {
      guard parts.graph_declared_upper_bound_bytes is Some(bound) &&
        parts.graph_declared_alignment_bytes is Some(alignment) &&
        parts.graph_descriptor_ceiling_bytes is Some(ceiling) &&
        parts.device_startup_upper_bound_bytes is Some(upper) else {
        raise InconsistentAdmission
      }
      if bound <= 0L ||
        alignment <= 0L ||
        (alignment & (alignment - 1L)) != 0L ||
        bound % alignment != 0L ||
        ceiling < bound ||
        checked_nonnegative_add(parts.graph_arena_reserved_total_bytes, bound) !=
        upper {
        raise InconsistentAdmission
      }
    }
  }
}

///|
fn runtime_report_from_parts(
  parts : RuntimeReportParts,
) -> RuntimeReport raise RuntimeReportError {
  validate_parts(parts)
  let device_reserved_total_bytes = checked_nonnegative_add(
    checked_nonnegative_add(
      parts.weight_reserved_bytes,
      parts.activation_workspace_reserved_bytes,
    ),
    parts.kv_reserved_bytes,
  )
  if device_reserved_total_bytes != parts.graph_arena_reserved_total_bytes {
    raise InconsistentAdmission
  }
  {
    weight_materialized_bytes: parts.weight_materialized_bytes,
    weight_reserved_bytes: parts.weight_reserved_bytes,
    weight_alignment_bytes: parts.weight_alignment_bytes,
    activation_reserved_high_water_bytes: parts.activation_reserved_high_water_bytes,
    workspace_offset_bytes: parts.workspace_offset_bytes,
    workspace_bytes: parts.workspace_bytes,
    workspace_alignment_bytes: parts.workspace_alignment_bytes,
    activation_workspace_reserved_bytes: parts.activation_workspace_reserved_bytes,
    kv_layout_version: parts.kv_layout_version,
    kv_tokens_per_page: parts.kv_tokens_per_page,
    kv_page_count: parts.kv_page_count,
    kv_page_logical_bytes: parts.kv_page_logical_bytes,
    kv_page_reserved_stride_bytes: parts.kv_page_reserved_stride_bytes,
    kv_component_stride_bytes: parts.kv_component_stride_bytes,
    kv_layer_stride_bytes: parts.kv_layer_stride_bytes,
    kv_segment_alignment_bytes: parts.kv_segment_alignment_bytes,
    kv_reserved_bytes: parts.kv_reserved_bytes,
    kernel_module_file_bytes: parts.kernel_module_file_bytes,
    device_reserved_total_bytes,
    graph_memory_availability: parts.graph_memory_availability,
    graph_declared_upper_bound_bytes: parts.graph_declared_upper_bound_bytes,
    graph_declared_alignment_bytes: parts.graph_declared_alignment_bytes,
    graph_descriptor_ceiling_bytes: parts.graph_descriptor_ceiling_bytes,
    device_startup_upper_bound_bytes: parts.device_startup_upper_bound_bytes,
    max_prefill_rows: parts.max_prefill_rows,
    max_decode_rows: parts.max_decode_rows,
    max_plan_rows: parts.max_plan_rows,
    max_plan_tokens: parts.max_plan_tokens,
    max_plan_pages: parts.max_plan_pages,
    max_completion_slots: parts.max_completion_slots,
    max_capabilities: parts.max_capabilities,
    max_sequence_tokens: parts.max_sequence_tokens,
    worker_max_token_id: parts.worker_max_token_id,
    max_input_tokens: parts.max_input_tokens,
    max_new_tokens: parts.max_new_tokens,
    max_context_tokens: parts.max_context_tokens,
    inference_max_token_id: parts.inference_max_token_id,
  }
}

///|
/// Builds a bounded operator report from inert, already-admitted descriptor
/// evidence. No filesystem access, device probe, allocation, or readiness
/// decision occurs here.
pub fn build(
  admission : @descriptor_file.RuntimeDescriptorAdmission,
) -> RuntimeReport raise RuntimeReportError {
  let execution = admission.execution()
  let weights = execution.weight_layout()
  let memory = execution.memory_plan()
  let workspace = memory.workspace()
  let kv = execution.layout()
  let startup = admission.startup_contract()
  let worker = startup.worker_limits()
  let inference = startup.inference_limits()
  let device = execution.device_step_limits()
  let graph = admission.graph_memory_capacity()
  runtime_report_from_parts({
    weight_materialized_bytes: weights.materialized_bytes(),
    weight_reserved_bytes: weights.arena_bytes(),
    weight_alignment_bytes: weights.alignment_bytes(),
    activation_reserved_high_water_bytes: memory.activation_bytes(),
    workspace_offset_bytes: workspace.offset_bytes(),
    workspace_bytes: workspace.byte_length(),
    workspace_alignment_bytes: workspace.alignment_bytes(),
    activation_workspace_reserved_bytes: memory.arena_bytes(),
    kv_layout_version: kv.stable_layout_version(),
    kv_tokens_per_page: kv.tokens_per_page(),
    kv_page_count: kv.total_page_count(),
    kv_page_logical_bytes: kv.page_logical_bytes(),
    kv_page_reserved_stride_bytes: kv.page_stride_bytes(),
    kv_component_stride_bytes: kv.component_stride_bytes(),
    kv_layer_stride_bytes: kv.layer_stride_bytes(),
    kv_segment_alignment_bytes: kv.segment_alignment_bytes(),
    kv_reserved_bytes: kv.arena_bytes(),
    kernel_module_file_bytes: execution.artifacts().total_module_bytes(),
    graph_memory_availability: graph.availability(),
    graph_declared_upper_bound_bytes: graph.declared_upper_bound_bytes(),
    graph_declared_alignment_bytes: graph.declared_alignment_bytes(),
    graph_descriptor_ceiling_bytes: graph.descriptor_ceiling_bytes(),
    graph_arena_reserved_total_bytes: graph.arena_reserved_total_bytes(),
    device_startup_upper_bound_bytes: graph.device_startup_upper_bound_bytes(),
    max_prefill_rows: worker.max_prefill_rows(),
    max_decode_rows: worker.max_decode_rows(),
    max_plan_rows: worker.max_plan_rows(),
    max_plan_tokens: worker.max_plan_tokens(),
    max_plan_pages: worker.max_plan_pages(),
    max_completion_slots: worker.max_completion_slots(),
    max_capabilities: worker.max_capabilities(),
    max_sequence_tokens: worker.max_sequence_tokens(),
    worker_max_token_id: worker.max_token_id(),
    max_input_tokens: inference.max_input_tokens(),
    max_new_tokens: inference.max_new_tokens(),
    max_context_tokens: inference.max_context_tokens(),
    inference_max_token_id: inference.max_token_id(),
    device_max_rows: device.max_rows(),
    device_max_tokens: device.max_tokens(),
    device_max_pages: device.max_pages(),
    device_max_sequence_tokens: device.max_sequence_tokens(),
    device_max_token_id: device.max_token_id(),
  })
}
