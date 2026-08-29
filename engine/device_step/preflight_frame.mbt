///|
fn wire_error(error : @worker_wire.WorkerWireError) -> DeviceStepError {
  Wire(error)
}

///|
fn validate_frame_counts(
  frame : @worker_wire.ValidatedPlanFrame,
  expected_generation : UInt64,
  layout : @device_layout.DeviceKvLayout,
  limits : DeviceStepLimits,
  expected_identity : @spec.ModelIdentity,
) -> DeviceStepSummary raise DeviceStepError {
  if layout.identity() != expected_identity ||
    layout.stable_layout_version() != 1 {
    raise InvalidLayout
  }
  let generation = frame.model_generation_value() catch {
    error => raise wire_error(error)
  }
  if generation != expected_generation {
    raise InvalidPlan(ModelGeneration)
  }
  let sequence_value = frame.sequence_value() catch {
    error => raise wire_error(error)
  }
  if sequence_value == 0UL {
    raise InvalidPlan(PlanSequence)
  }
  let prefill_count = frame.prefill_count() catch {
    error => raise wire_error(error)
  }
  let decode_count = frame.decode_count() catch {
    error => raise wire_error(error)
  }
  let row_count = prefill_count + decode_count
  let token_count = frame.token_count() catch {
    error => raise wire_error(error)
  }
  let page_count = frame.page_count() catch { error => raise wire_error(error) }
  let token_budget = frame.token_budget() catch {
    error => raise wire_error(error)
  }
  if row_count <= 0 ||
    row_count > limits.max_rows() ||
    token_count <= 0 ||
    token_count > limits.max_tokens() ||
    page_count <= 0 ||
    page_count > limits.max_pages() ||
    token_budget != token_count {
    raise InvalidPlan(Counts)
  }
  {
    sequence_value,
    prefill_count,
    decode_count,
    row_count,
    token_count,
    page_count,
  }
}

///|
fn write_frame_tokens(
  frame : @worker_wire.ValidatedPlanFrame,
  host : DeviceStepHostBuffers,
  limits : DeviceStepLimits,
  token_start : Int,
  token_length : Int,
  sequence_start : Int,
  write : Bool,
) -> Unit raise DeviceStepError {
  for local_index in 0..<token_length {
    let table_index = token_start + local_index
    let token = frame.token_at(table_index) catch {
      error => raise wire_error(error)
    }
    if token < 0 || token > limits.max_token_id() {
      raise InvalidPlan(TokenId)
    }
    let position = checked_i32(sequence_start + local_index, TokenPosition)
    if write {
      write_i32(host.token_ids, table_index, token)
      write_i32(host.token_positions, table_index, position)
    }
  }
}

///|
fn write_frame_pages(
  frame : @worker_wire.ValidatedPlanFrame,
  host : DeviceStepHostBuffers,
  page_start : Int,
  page_length : Int,
  plan_page_count : Int,
  physical_page_capacity : Int,
  write : Bool,
) -> Unit raise DeviceStepError {
  if page_start < 0 ||
    page_length <= 0 ||
    page_start > plan_page_count - page_length {
    raise InvalidPlan(PagePacking)
  }
  for local_index in 0..<page_length {
    let table_index = page_start + local_index
    let generation = frame.page_generation_at(table_index) catch {
      error => raise wire_error(error)
    }
    let page_index = frame.page_index_at(table_index) catch {
      error => raise wire_error(error)
    }
    if generation <= 0 {
      raise InvalidPlan(PageIdStructure)
    }
    if page_index < 0 || page_index >= physical_page_capacity {
      raise InvalidPlan(PageIndex)
    }
    checked_i32(page_index, PageIndex) |> ignore
    if write {
      write_i32(host.physical_page_indices, table_index, page_index)
    }
  }
}

///|
fn validate_frame_capabilities(
  frame : @worker_wire.ValidatedPlanFrame,
  start : Int,
  length : Int,
  required : FixedArray[@plan.KernelCapabilityId],
) -> Unit raise DeviceStepError {
  if length != required.length() {
    raise InvalidPlan(CapabilityRecipe)
  }
  for index in 0..<required.length() {
    let actual = frame.capability_value_at(start + index) catch {
      error => raise wire_error(error)
    }
    if actual != required[index].as_int() {
      raise InvalidPlan(CapabilityRecipe)
    }
  }
}

///|
fn write_frame_prefill(
  frame : @worker_wire.ValidatedPlanFrame,
  host : DeviceStepHostBuffers,
  row_index : Int,
  expected_token_start : Int,
  expected_page_start : Int,
  summary : DeviceStepSummary,
  layout : @device_layout.DeviceKvLayout,
  limits : DeviceStepLimits,
  required : FixedArray[@plan.KernelCapabilityId],
  greedy_only : Bool,
  write : Bool,
) -> Int64 raise DeviceStepError {
  let row = frame.prefill_row(row_index) catch {
    error => raise wire_error(error)
  }
  let token_start = row.token_start() catch { error => raise wire_error(error) }
  let token_length = row.token_length() catch {
    error => raise wire_error(error)
  }
  let page_start = row.page_start() catch { error => raise wire_error(error) }
  let page_length = row.page_length() catch { error => raise wire_error(error) }
  let capability_start = row.capability_start() catch {
    error => raise wire_error(error)
  }
  let capability_length = row.capability_length() catch {
    error => raise wire_error(error)
  }
  if greedy_only &&
    (row.produces_token() catch { error => raise wire_error(error) }) &&
    (row.sampling_mode_value() catch { error => raise wire_error(error) }) != 0 {
    raise InvalidExecutor(SamplingMode)
  }
  validate_frame_capabilities(
    frame, capability_start, capability_length, required,
  )
  let sequence_start = row.sequence_token_start() catch {
    error => raise wire_error(error)
  }
  if token_start != expected_token_start || token_length <= 0 {
    raise InvalidPlan(TokenPacking)
  }
  if page_start != expected_page_start {
    raise InvalidPlan(PagePacking)
  }
  let sequence_length = checked_i32(
    sequence_start + token_length,
    SequenceLength,
  )
  if sequence_length > limits.max_sequence_tokens() {
    raise InvalidPlan(SequenceLength)
  }
  if page_length != expected_pages(sequence_length, layout.tokens_per_page()) {
    raise InvalidPlan(PageCount)
  }
  write_frame_tokens(
    frame, host, limits, token_start, token_length, sequence_start, write,
  )
  write_frame_pages(
    frame,
    host,
    page_start,
    page_length,
    summary.page_count,
    layout.total_page_count(),
    write,
  )
  if write {
    write_i32(host.query_row_offsets, row_index, token_start)
    write_i32(host.sequence_lengths, row_index, sequence_length)
    write_i32(host.page_offsets, row_index, page_start)
  }
  (Int64::from_int(token_start + token_length) << 32) |
  Int64::from_int(page_start + page_length)
}

///|
fn write_frame_decode(
  frame : @worker_wire.ValidatedPlanFrame,
  host : DeviceStepHostBuffers,
  decode_index : Int,
  row_index : Int,
  expected_token_start : Int,
  expected_page_start : Int,
  summary : DeviceStepSummary,
  layout : @device_layout.DeviceKvLayout,
  limits : DeviceStepLimits,
  required : FixedArray[@plan.KernelCapabilityId],
  greedy_only : Bool,
  write : Bool,
) -> Int64 raise DeviceStepError {
  let row = frame.decode_row(decode_index) catch {
    error => raise wire_error(error)
  }
  let token_start = row.token_start() catch { error => raise wire_error(error) }
  let page_start = row.page_start() catch { error => raise wire_error(error) }
  let page_length = row.page_length() catch { error => raise wire_error(error) }
  let capability_start = row.capability_start() catch {
    error => raise wire_error(error)
  }
  let capability_length = row.capability_length() catch {
    error => raise wire_error(error)
  }
  if greedy_only &&
    (row.sampling_mode_value() catch { error => raise wire_error(error) }) != 0 {
    raise InvalidExecutor(SamplingMode)
  }
  validate_frame_capabilities(
    frame, capability_start, capability_length, required,
  )
  let position = row.sequence_token_position() catch {
    error => raise wire_error(error)
  }
  if token_start != expected_token_start {
    raise InvalidPlan(TokenPacking)
  }
  if page_start != expected_page_start {
    raise InvalidPlan(PagePacking)
  }
  checked_i32(position, TokenPosition) |> ignore
  let sequence_length = checked_i32(position + 1, SequenceLength)
  if sequence_length > limits.max_sequence_tokens() {
    raise InvalidPlan(SequenceLength)
  }
  if page_length != expected_pages(sequence_length, layout.tokens_per_page()) {
    raise InvalidPlan(PageCount)
  }
  write_frame_tokens(frame, host, limits, token_start, 1, position, write)
  write_frame_pages(
    frame,
    host,
    page_start,
    page_length,
    summary.page_count,
    layout.total_page_count(),
    write,
  )
  if write {
    write_i32(host.query_row_offsets, row_index, token_start)
    write_i32(host.sequence_lengths, row_index, sequence_length)
    write_i32(host.page_offsets, row_index, page_start)
  }
  (Int64::from_int(token_start + 1) << 32) |
  Int64::from_int(page_start + page_length)
}

///|
fn preflight_frame_and_write(
  frame : @worker_wire.ValidatedPlanFrame,
  expected_generation : UInt64,
  expected_identity : @spec.ModelIdentity,
  layout : @device_layout.DeviceKvLayout,
  limits : DeviceStepLimits,
  host : DeviceStepHostBuffers,
  required : FixedArray[@plan.KernelCapabilityId],
  write : Bool,
  greedy_only? : Bool = false,
) -> DeviceStepSummary raise DeviceStepError {
  let summary = validate_frame_counts(
    frame, expected_generation, layout, limits, expected_identity,
  )
  let mut token_end = 0
  let mut page_end = 0
  for index in 0..<summary.prefill_count {
    let cursor = write_frame_prefill(
      frame, host, index, token_end, page_end, summary, layout, limits, required,
      greedy_only, write,
    )
    token_end = cursor_token_end(cursor)
    page_end = cursor_page_end(cursor)
  }
  for index in 0..<summary.decode_count {
    let cursor = write_frame_decode(
      frame,
      host,
      index,
      summary.prefill_count + index,
      token_end,
      page_end,
      summary,
      layout,
      limits,
      required,
      greedy_only,
      write,
    )
    token_end = cursor_token_end(cursor)
    page_end = cursor_page_end(cursor)
  }
  if token_end != summary.token_count {
    raise InvalidPlan(TokenPacking)
  }
  if page_end != summary.page_count {
    raise InvalidPlan(PagePacking)
  }
  if write {
    write_i32(host.query_row_offsets, summary.row_count, summary.token_count)
    write_i32(host.page_offsets, summary.row_count, summary.page_count)
    write_i32(host.counts, 0, summary.prefill_count)
    write_i32(host.counts, 1, summary.decode_count)
    write_i32(host.counts, 2, summary.row_count)
    write_i32(host.counts, 3, summary.token_count)
    write_i32(host.counts, 4, summary.page_count)
  }
  summary
}
