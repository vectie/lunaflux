///|
fn append_report_line(
  output : StringBuilder,
  label : String,
  value : Int64,
) -> Unit {
  output.write_string(label)
  output.write_string(": ")
  output.write_string(value.to_string())
  output.write_string("\n")
}

///|
fn append_report_int(
  output : StringBuilder,
  label : String,
  value : Int,
) -> Unit {
  output.write_string(label)
  output.write_string(": ")
  output.write_string(value.to_string())
  output.write_string("\n")
}

///|
fn append_graph_memory_report(
  output : StringBuilder,
  report : RuntimeReport,
) -> Unit {
  match report.graph_memory_availability {
    GraphAccountingAbsent =>
      output.write_string(
        "memory.graph_capture.accounting: absent (runtime descriptor does not admit graph capture)\n",
      )
    GraphAccountingUnsupported =>
      output.write_string(
        "memory.graph_capture.accounting: unsupported (authenticated kernel metadata rejects capture)\n",
      )
    GraphAccountingBounded => {
      output.write_string(
        "memory.graph_capture.accounting: declared_upper_bound\n",
      )
      guard report.graph_declared_upper_bound_bytes is Some(bound) &&
        report.graph_declared_alignment_bytes is Some(alignment) &&
        report.graph_descriptor_ceiling_bytes is Some(ceiling) &&
        report.device_startup_upper_bound_bytes is Some(upper) else {
        abort("invalid admitted graph-memory report")
      }
      append_report_line(
        output, "memory.graph_capture.declared_upper_bound_bytes", bound,
      )
      append_report_line(
        output, "memory.graph_capture.alignment_bytes", alignment,
      )
      append_report_line(
        output, "memory.graph_capture.descriptor_ceiling_bytes", ceiling,
      )
      append_report_line(
        output, "memory.device.startup_upper_bound_bytes", upper,
      )
    }
  }
}

///|
/// Renders a fixed-order, bounded operator report. Artifact file bytes are
/// explicitly separate from the device-allocation total. Scheduler and
/// service capacities remain unavailable because descriptor v1 admits neither
/// SchedulerConfig nor CacheConfig nor service configuration.
fn render_runtime(report : RuntimeReport, unavailable : Bool) -> String {
  let output = StringBuilder()
  output.write_string("LunaFlux admitted runtime memory and capacity\n")
  append_report_line(
    output,
    "memory.weight.materialized_bytes",
    report.weight_materialized_bytes,
  )
  append_report_line(
    output,
    "memory.weight.reserved_arena_bytes",
    report.weight_reserved_bytes,
  )
  append_report_line(
    output,
    "memory.weight.alignment_bytes",
    report.weight_alignment_bytes,
  )
  append_report_line(
    output,
    "memory.activation.reserved_high_water_bytes",
    report.activation_reserved_high_water_bytes,
  )
  append_report_line(
    output,
    "memory.workspace.offset_bytes",
    report.workspace_offset_bytes,
  )
  append_report_line(output, "memory.workspace.bytes", report.workspace_bytes)
  append_report_line(
    output,
    "memory.workspace.alignment_bytes",
    report.workspace_alignment_bytes,
  )
  append_report_line(
    output,
    "memory.activation_workspace.reserved_arena_bytes",
    report.activation_workspace_reserved_bytes,
  )
  append_report_int(
    output,
    "memory.kv.layout_version",
    report.kv_layout_version,
  )
  append_report_int(
    output,
    "memory.kv.tokens_per_page",
    report.kv_tokens_per_page,
  )
  append_report_int(output, "memory.kv.page_count", report.kv_page_count)
  append_report_line(
    output,
    "memory.kv.page_logical_bytes",
    report.kv_page_logical_bytes,
  )
  append_report_line(
    output,
    "memory.kv.page_reserved_stride_bytes",
    report.kv_page_reserved_stride_bytes,
  )
  append_report_line(
    output,
    "memory.kv.component_stride_bytes",
    report.kv_component_stride_bytes,
  )
  append_report_line(
    output,
    "memory.kv.layer_stride_bytes",
    report.kv_layer_stride_bytes,
  )
  append_report_line(
    output,
    "memory.kv.segment_alignment_bytes",
    report.kv_segment_alignment_bytes,
  )
  append_report_line(
    output,
    "memory.kv.reserved_arena_bytes",
    report.kv_reserved_bytes,
  )
  append_report_line(
    output,
    "artifacts.kernel_module_file_bytes_excluded_from_device_total",
    report.kernel_module_file_bytes,
  )
  append_report_line(
    output,
    "memory.device.reserved_total_bytes",
    report.device_reserved_total_bytes,
  )
  append_graph_memory_report(output, report)
  append_report_int(
    output,
    "execution.max_prefill_rows",
    report.max_prefill_rows,
  )
  append_report_int(output, "execution.max_decode_rows", report.max_decode_rows)
  append_report_int(output, "execution.max_plan_rows", report.max_plan_rows)
  append_report_int(output, "execution.max_plan_tokens", report.max_plan_tokens)
  append_report_int(output, "execution.max_plan_pages", report.max_plan_pages)
  append_report_int(
    output,
    "execution.max_completion_slots",
    report.max_completion_slots,
  )
  append_report_int(
    output,
    "execution.max_capabilities",
    report.max_capabilities,
  )
  append_report_int(
    output,
    "execution.max_sequence_tokens",
    report.max_sequence_tokens,
  )
  append_report_int(
    output,
    "execution.max_token_id",
    report.worker_max_token_id,
  )
  append_report_int(
    output,
    "inference.max_input_tokens",
    report.max_input_tokens,
  )
  append_report_int(output, "inference.max_new_tokens", report.max_new_tokens)
  append_report_int(
    output,
    "inference.max_context_tokens",
    report.max_context_tokens,
  )
  append_report_int(
    output,
    "inference.max_token_id",
    report.inference_max_token_id,
  )
  if unavailable {
    output.write_string(
      "scheduler.capacity: unavailable (descriptor v1 does not admit SchedulerConfig or CacheConfig)\n",
    )
    output.write_string(
      "service.capacity: unavailable (descriptor v1 does not admit service configuration)\n",
    )
  }
  output.to_string()
}

///|
/// Render descriptor-only evidence and explicitly mark missing policy data.
pub fn render(report : RuntimeReport) -> String {
  render_runtime(report, true)
}

///|
/// Render fixed-order runtime plus policy-backed instance capacities.
pub fn render_instance(report : RuntimeInstanceReport) -> String {
  let output = StringBuilder()
  output.write_string(render_runtime(report.runtime, false))
  append_report_int(
    output,
    "tokenizer.vocabulary_size",
    report.tokenizer_vocabulary_size,
  )
  append_report_int(
    output,
    "scheduler.total_request_slots",
    report.scheduler_total_request_slots,
  )
  append_report_int(
    output,
    "scheduler.max_active_requests",
    report.scheduler_max_active_requests,
  )
  append_report_int(
    output,
    "scheduler.max_waiting_requests",
    report.scheduler_max_waiting_requests,
  )
  append_report_int(
    output,
    "scheduler.max_rows_per_step",
    report.scheduler_max_rows_per_step,
  )
  output.write_string("scheduler.prefix_enabled: ")
  output.write_string(
    if report.scheduler_prefix_enabled {
      "true"
    } else {
      "false"
    },
  )
  output.write_string("\n")
  append_report_int(
    output,
    "service.max_request_bytes",
    report.service_max_request_bytes,
  )
  append_report_int(
    output,
    "preparation.lane_count",
    report.preparation_lane_count,
  )
  append_report_int(
    output,
    "telemetry.instance_log_capacity",
    report.telemetry_capacity,
  )
  output.write_string("service.openai_activation: not admitted\n")
  output.to_string()
}
