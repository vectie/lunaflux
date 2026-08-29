///|
fn protocol_error(
  error : @worker_protocol.WorkerProtocolError,
) -> DeviceStepError {
  Protocol(error)
}

///|
fn checked_i32(
  value : Int,
  rule : DeviceStepPlanRule,
) -> Int raise DeviceStepError {
  if value < 0 || value > INT32_MAX {
    raise InvalidPlan(rule)
  }
  value
}

///|
fn expected_pages(
  sequence_length : Int,
  tokens_per_page : Int,
) -> Int raise DeviceStepError {
  if sequence_length <= 0 || sequence_length > INT32_MAX {
    raise InvalidPlan(SequenceLength)
  }
  1 + (sequence_length - 1) / tokens_per_page
}

///|
fn validate_identity_and_counts(
  plan : @worker_protocol.SubmittedSchedulePlan,
  expected_generation : UInt64,
  layout : @device_layout.DeviceKvLayout,
  limits : DeviceStepLimits,
  expected_identity : @spec.ModelIdentity,
) -> DeviceStepSummary raise DeviceStepError {
  if layout.identity() != expected_identity ||
    layout.stable_layout_version() != 1 {
    raise InvalidLayout
  }
  let generation = plan.model_plan_generation_wire() catch {
    error => raise protocol_error(error)
  }
  if generation != expected_generation {
    raise InvalidPlan(ModelGeneration)
  }
  let sequence_value = plan.sequence_wire() catch {
    error => raise protocol_error(error)
  }
  if sequence_value == 0UL {
    raise InvalidPlan(PlanSequence)
  }
  let prefill_count = plan.prefill_count() catch {
    error => raise protocol_error(error)
  }
  let decode_count = plan.decode_count() catch {
    error => raise protocol_error(error)
  }
  let row_count = plan.row_count() catch {
    error => raise protocol_error(error)
  }
  let token_count = plan.token_count() catch {
    error => raise protocol_error(error)
  }
  let page_count = plan.page_count() catch {
    error => raise protocol_error(error)
  }
  let token_budget = plan.token_budget() catch {
    error => raise protocol_error(error)
  }
  if row_count != prefill_count + decode_count ||
    row_count <= 0 ||
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
fn write_tokens_and_positions(
  plan : @worker_protocol.SubmittedSchedulePlan,
  host : DeviceStepHostBuffers,
  limits : DeviceStepLimits,
  token_start : Int,
  token_length : Int,
  sequence_start : Int,
  write : Bool,
) -> Unit raise DeviceStepError {
  for local_index in 0..<token_length {
    let table_index = token_start + local_index
    let token = plan.token_at(table_index) catch {
      error => raise protocol_error(error)
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
fn write_pages(
  plan : @worker_protocol.SubmittedSchedulePlan,
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
    let page = plan.page_at(table_index) catch {
      error => raise protocol_error(error)
    }
    if page.generation() <= 0 {
      raise InvalidPlan(PageIdStructure)
    }
    if page.index() < 0 || page.index() >= physical_page_capacity {
      raise InvalidPlan(PageIndex)
    }
    checked_i32(page.index(), PageIndex) |> ignore
    if write {
      write_i32(host.physical_page_indices, table_index, page.index())
    }
  }
}

///|
fn validate_capabilities(
  plan : @worker_protocol.SubmittedSchedulePlan,
  view : @worker_protocol.CapabilityView,
  required : FixedArray[@plan.KernelCapabilityId],
) -> Unit raise DeviceStepError {
  if view.length() != required.length() {
    raise InvalidPlan(CapabilityRecipe)
  }
  for index in 0..<required.length() {
    let actual = plan.capability_at(view.start() + index) catch {
      error => raise protocol_error(error)
    }
    if actual != required[index] {
      raise InvalidPlan(CapabilityRecipe)
    }
  }
}

///|
fn write_prefill_row(
  plan : @worker_protocol.SubmittedSchedulePlan,
  host : DeviceStepHostBuffers,
  row_index : Int,
  expected_token_start : Int,
  expected_page_start : Int,
  plan_page_count : Int,
  physical_page_capacity : Int,
  tokens_per_page : Int,
  limits : DeviceStepLimits,
  required_capabilities : FixedArray[@plan.KernelCapabilityId],
  greedy_only : Bool,
  write : Bool,
) -> Int64 raise DeviceStepError {
  let row = plan.prefill_row(row_index) catch {
    error => raise protocol_error(error)
  }
  let tokens = row.tokens() catch { error => raise protocol_error(error) }
  let pages = row.page_table() catch { error => raise protocol_error(error) }
  let capabilities = row.capabilities() catch {
    error => raise protocol_error(error)
  }
  if greedy_only &&
    (row.produces_token() catch { error => raise protocol_error(error) }) &&
    (row.sampling() catch { error => raise protocol_error(error) }).mode() !=
    Greedy {
    raise InvalidExecutor(SamplingMode)
  }
  validate_capabilities(plan, capabilities, required_capabilities)
  let sequence_start = row.sequence_token_start() catch {
    error => raise protocol_error(error)
  }
  if tokens.start() != expected_token_start || tokens.length() <= 0 {
    raise InvalidPlan(TokenPacking)
  }
  if pages.start() != expected_page_start {
    raise InvalidPlan(PagePacking)
  }
  let sequence_length = checked_i32(
    sequence_start + tokens.length(),
    SequenceLength,
  )
  if sequence_length > limits.max_sequence_tokens() {
    raise InvalidPlan(SequenceLength)
  }
  if pages.length() != expected_pages(sequence_length, tokens_per_page) {
    raise InvalidPlan(PageCount)
  }
  write_tokens_and_positions(
    plan,
    host,
    limits,
    tokens.start(),
    tokens.length(),
    sequence_start,
    write,
  )
  write_pages(
    plan,
    host,
    pages.start(),
    pages.length(),
    plan_page_count,
    physical_page_capacity,
    write,
  )
  if write {
    write_i32(host.query_row_offsets, row_index, tokens.start())
    write_i32(host.sequence_lengths, row_index, sequence_length)
    write_i32(host.page_offsets, row_index, pages.start())
  }
  (Int64::from_int(tokens.end()) << 32) | Int64::from_int(pages.end())
}

///|
fn write_decode_row(
  plan : @worker_protocol.SubmittedSchedulePlan,
  host : DeviceStepHostBuffers,
  decode_index : Int,
  row_index : Int,
  expected_token_start : Int,
  expected_page_start : Int,
  plan_page_count : Int,
  physical_page_capacity : Int,
  tokens_per_page : Int,
  limits : DeviceStepLimits,
  required_capabilities : FixedArray[@plan.KernelCapabilityId],
  greedy_only : Bool,
  write : Bool,
) -> Int64 raise DeviceStepError {
  let row = plan.decode_row(decode_index) catch {
    error => raise protocol_error(error)
  }
  let tokens = row.input_token() catch { error => raise protocol_error(error) }
  let pages = row.page_table() catch { error => raise protocol_error(error) }
  let capabilities = row.capabilities() catch {
    error => raise protocol_error(error)
  }
  if greedy_only &&
    (row.sampling() catch { error => raise protocol_error(error) }).mode() !=
    Greedy {
    raise InvalidExecutor(SamplingMode)
  }
  validate_capabilities(plan, capabilities, required_capabilities)
  let position = row.sequence_token_position() catch {
    error => raise protocol_error(error)
  }
  if tokens.start() != expected_token_start || tokens.length() != 1 {
    raise InvalidPlan(TokenPacking)
  }
  if pages.start() != expected_page_start {
    raise InvalidPlan(PagePacking)
  }
  checked_i32(position, TokenPosition) |> ignore
  let sequence_length = checked_i32(position + 1, SequenceLength)
  if sequence_length > limits.max_sequence_tokens() {
    raise InvalidPlan(SequenceLength)
  }
  if pages.length() != expected_pages(sequence_length, tokens_per_page) {
    raise InvalidPlan(PageCount)
  }
  write_tokens_and_positions(
    plan,
    host,
    limits,
    tokens.start(),
    1,
    position,
    write,
  )
  write_pages(
    plan,
    host,
    pages.start(),
    pages.length(),
    plan_page_count,
    physical_page_capacity,
    write,
  )
  if write {
    write_i32(host.query_row_offsets, row_index, tokens.start())
    write_i32(host.sequence_lengths, row_index, sequence_length)
    write_i32(host.page_offsets, row_index, pages.start())
  }
  (Int64::from_int(tokens.end()) << 32) | Int64::from_int(pages.end())
}

///|
fn cursor_token_end(cursor : Int64) -> Int {
  (cursor >> 32).to_int()
}

///|
fn cursor_page_end(cursor : Int64) -> Int {
  (cursor & 0xffffffffL).to_int()
}

///|
fn preflight_and_write(
  plan : @worker_protocol.SubmittedSchedulePlan,
  expected_generation : UInt64,
  expected_identity : @spec.ModelIdentity,
  layout : @device_layout.DeviceKvLayout,
  limits : DeviceStepLimits,
  host : DeviceStepHostBuffers,
  required_capabilities : FixedArray[@plan.KernelCapabilityId],
  write : Bool,
  greedy_only? : Bool = false,
) -> DeviceStepSummary raise DeviceStepError {
  let summary = validate_identity_and_counts(
    plan, expected_generation, layout, limits, expected_identity,
  )
  let mut token_end = 0
  let mut page_end = 0
  for prefill_index in 0..<summary.prefill_count {
    let cursor = write_prefill_row(
      plan,
      host,
      prefill_index,
      token_end,
      page_end,
      summary.page_count,
      layout.total_page_count(),
      layout.tokens_per_page(),
      limits,
      required_capabilities,
      greedy_only,
      write,
    )
    token_end = cursor_token_end(cursor)
    page_end = cursor_page_end(cursor)
  }
  for decode_index in 0..<summary.decode_count {
    let row_index = summary.prefill_count + decode_index
    let cursor = write_decode_row(
      plan,
      host,
      decode_index,
      row_index,
      token_end,
      page_end,
      summary.page_count,
      layout.total_page_count(),
      layout.tokens_per_page(),
      limits,
      required_capabilities,
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
