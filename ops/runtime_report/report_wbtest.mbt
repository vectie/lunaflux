///|
fn report_parts(
  weight_materialized_bytes? : Int64 = 9L,
  weight_reserved_bytes? : Int64 = 10L,
  activation_reserved_high_water_bytes? : Int64 = 12L,
  workspace_offset_bytes? : Int64 = 12L,
  workspace_bytes? : Int64 = 8L,
  activation_workspace_reserved_bytes? : Int64 = 20L,
  kv_page_logical_bytes? : Int64 = 24L,
  kv_page_reserved_stride_bytes? : Int64 = 32L,
  kv_reserved_bytes? : Int64 = 1024L,
  kernel_module_file_bytes? : Int64 = 40L,
  graph_memory_availability? : @runtime_resolved.RuntimeGraphMemoryAvailability = GraphAccountingAbsent,
  graph_declared_upper_bound_bytes? : Int64,
  graph_declared_alignment_bytes? : Int64,
  graph_descriptor_ceiling_bytes? : Int64,
  graph_arena_reserved_total_bytes? : Int64 = 1054L,
  device_startup_upper_bound_bytes? : Int64,
  max_plan_rows? : Int = 4,
  device_max_rows? : Int = 4,
) -> RuntimeReportParts {
  {
    weight_materialized_bytes,
    weight_reserved_bytes,
    weight_alignment_bytes: 256L,
    activation_reserved_high_water_bytes,
    workspace_offset_bytes,
    workspace_bytes,
    workspace_alignment_bytes: 256L,
    activation_workspace_reserved_bytes,
    kv_layout_version: 1,
    kv_tokens_per_page: 16,
    kv_page_count: 8,
    kv_page_logical_bytes,
    kv_page_reserved_stride_bytes,
    kv_component_stride_bytes: 256L,
    kv_layer_stride_bytes: 512L,
    kv_segment_alignment_bytes: 256L,
    kv_reserved_bytes,
    kernel_module_file_bytes,
    graph_memory_availability,
    graph_declared_upper_bound_bytes,
    graph_declared_alignment_bytes,
    graph_descriptor_ceiling_bytes,
    graph_arena_reserved_total_bytes,
    device_startup_upper_bound_bytes,
    max_prefill_rows: 2,
    max_decode_rows: 4,
    max_plan_rows,
    max_plan_tokens: 64,
    max_plan_pages: 16,
    max_completion_slots: 8,
    max_capabilities: 32,
    max_sequence_tokens: 128,
    worker_max_token_id: 255,
    max_input_tokens: 64,
    max_new_tokens: 32,
    max_context_tokens: 128,
    inference_max_token_id: 127,
    device_max_rows,
    device_max_tokens: 64,
    device_max_pages: 16,
    device_max_sequence_tokens: 128,
    device_max_token_id: 255,
  }
}

///|
test "runtime report is fixed-order exact and excludes workspace and artifacts from total" {
  let report = runtime_report_from_parts(report_parts())
  assert_eq(report.device_reserved_total_bytes, 1054L)
  assert_eq(
    render(report),
    (
      #|LunaFlux admitted runtime memory and capacity
      #|memory.weight.materialized_bytes: 9
      #|memory.weight.reserved_arena_bytes: 10
      #|memory.weight.alignment_bytes: 256
      #|memory.activation.reserved_high_water_bytes: 12
      #|memory.workspace.offset_bytes: 12
      #|memory.workspace.bytes: 8
      #|memory.workspace.alignment_bytes: 256
      #|memory.activation_workspace.reserved_arena_bytes: 20
      #|memory.kv.layout_version: 1
      #|memory.kv.tokens_per_page: 16
      #|memory.kv.page_count: 8
      #|memory.kv.page_logical_bytes: 24
      #|memory.kv.page_reserved_stride_bytes: 32
      #|memory.kv.component_stride_bytes: 256
      #|memory.kv.layer_stride_bytes: 512
      #|memory.kv.segment_alignment_bytes: 256
      #|memory.kv.reserved_arena_bytes: 1024
      #|artifacts.kernel_module_file_bytes_excluded_from_device_total: 40
      #|memory.device.reserved_total_bytes: 1054
      #|memory.graph_capture.accounting: absent (runtime descriptor does not admit graph capture)
      #|execution.max_prefill_rows: 2
      #|execution.max_decode_rows: 4
      #|execution.max_plan_rows: 4
      #|execution.max_plan_tokens: 64
      #|execution.max_plan_pages: 16
      #|execution.max_completion_slots: 8
      #|execution.max_capabilities: 32
      #|execution.max_sequence_tokens: 128
      #|execution.max_token_id: 255
      #|inference.max_input_tokens: 64
      #|inference.max_new_tokens: 32
      #|inference.max_context_tokens: 128
      #|inference.max_token_id: 127
      #|scheduler.capacity: unavailable (descriptor v1 does not admit SchedulerConfig or CacheConfig)
      #|service.capacity: unavailable (descriptor v1 does not admit service configuration)
      #|
    ),
  )
}

///|
test "runtime report rejects negative overlapping and inconsistent evidence" {
  let invalid_cases = [
    report_parts(weight_materialized_bytes=-1L),
    report_parts(weight_materialized_bytes=11L),
    report_parts(activation_reserved_high_water_bytes=21L),
    report_parts(workspace_offset_bytes=19L, workspace_bytes=2L),
    report_parts(kv_page_logical_bytes=33L),
  ]
  for parts in invalid_cases {
    try runtime_report_from_parts(parts) catch {
      InvalidAdmission => ()
      _ => fail("unexpected invalid report error")
    } noraise {
      _ => fail("expected invalid report evidence")
    }
  }
  try runtime_report_from_parts(report_parts(device_max_rows=3)) catch {
    InconsistentAdmission => ()
    _ => fail("unexpected inconsistent report error")
  } noraise {
    _ => fail("expected duplicated ceiling mismatch")
  }
}

///|
test "runtime report checked additions reject hostile Int64 overflow" {
  try
    runtime_report_from_parts(
      report_parts(
        weight_materialized_bytes=0x7fffffffffffffffL,
        weight_reserved_bytes=0x7fffffffffffffffL,
        activation_reserved_high_water_bytes=0L,
        workspace_offset_bytes=0L,
        workspace_bytes=0L,
        activation_workspace_reserved_bytes=1L,
        kv_reserved_bytes=512L,
        graph_arena_reserved_total_bytes=0L,
      ),
    )
  catch {
    DeviceBytesOverflow => ()
    _ => fail("unexpected total overflow error")
  } noraise {
    _ => fail("expected checked device-total overflow")
  }
  try
    runtime_report_from_parts(
      report_parts(
        workspace_offset_bytes=0x7fffffffffffffffL,
        workspace_bytes=1L,
        activation_workspace_reserved_bytes=0x7fffffffffffffffL,
      ),
    )
  catch {
    DeviceBytesOverflow => ()
    _ => fail("unexpected workspace overflow error")
  } noraise {
    _ => fail("expected checked workspace-end overflow")
  }
}

///|
test "bounded graph report labels a declared upper bound separately from arenas" {
  let report = runtime_report_from_parts(
    report_parts(
      graph_memory_availability=GraphAccountingBounded,
      graph_declared_upper_bound_bytes=4096L,
      graph_declared_alignment_bytes=256L,
      graph_descriptor_ceiling_bytes=8192L,
      device_startup_upper_bound_bytes=5150L,
    ),
  )
  let rendered = render(report)
  assert_true(
    rendered.contains("memory.graph_capture.accounting: declared_upper_bound\n"),
  )
  assert_true(
    rendered.contains("memory.graph_capture.declared_upper_bound_bytes: 4096\n"),
  )
  assert_true(
    rendered.contains("memory.graph_capture.descriptor_ceiling_bytes: 8192\n"),
  )
  assert_true(rendered.contains("memory.device.reserved_total_bytes: 1054\n"))
  assert_true(
    rendered.contains("memory.device.startup_upper_bound_bytes: 5150\n"),
  )
}

///|
test "graph report rejects partial and mismatched bounded projections" {
  let hostile = [
    report_parts(
      graph_memory_availability=GraphAccountingBounded,
      graph_declared_upper_bound_bytes=4096L,
      graph_declared_alignment_bytes=256L,
      graph_descriptor_ceiling_bytes=8192L,
    ),
    report_parts(
      graph_memory_availability=GraphAccountingBounded,
      graph_declared_upper_bound_bytes=4096L,
      graph_declared_alignment_bytes=256L,
      graph_descriptor_ceiling_bytes=2048L,
      device_startup_upper_bound_bytes=5150L,
    ),
    report_parts(
      graph_memory_availability=GraphAccountingAbsent,
      graph_declared_upper_bound_bytes=4096L,
    ),
  ]
  for parts in hostile {
    try runtime_report_from_parts(parts) catch {
      InconsistentAdmission => ()
      _ => fail("unexpected graph report rejection")
    } noraise {
      _ => fail("hostile graph report was admitted")
    }
  }
}

///|
test "typed instance report replaces descriptor-only unavailable labels" {
  let instance : RuntimeInstanceReport = {
    runtime: try! runtime_report_from_parts(report_parts()),
    tokenizer_vocabulary_size: 256,
    scheduler_total_request_slots: 8,
    scheduler_max_active_requests: 4,
    scheduler_max_waiting_requests: 4,
    scheduler_max_rows_per_step: 4,
    scheduler_prefix_enabled: true,
    service_max_request_bytes: 4096,
    preparation_lane_count: 8,
    telemetry_capacity: 1024,
  }
  let rendered = render_instance(instance)
  assert_true(rendered.contains("scheduler.total_request_slots: 8"))
  assert_true(rendered.contains("service.max_request_bytes: 4096"))
  assert_true(rendered.contains("service.openai_activation: not admitted"))
  assert_false(rendered.contains("scheduler.capacity: unavailable"))
}
