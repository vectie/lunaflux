///|
fn authenticate_batch_prefill(
  cycle : PreparedBatchCycle,
  index : Int,
  token_start : Int,
  token_length : Int,
  sequence_start : Int,
  stochastic : Bool,
) -> Unit raise {
  let row = cycle.frame.prefill_row(index)
  let base = cycle.frame.sequence_value() * 16UL
  let seed = base + (101 + index - 1).to_uint64()
  let sample_index = ((cycle.frame.sequence_value() + index.to_uint64() - 1UL) %
  8UL).to_int()
  if row.request_id_value() != cycle.expected_ids[index] ||
    row.request_generation_value() != cycle.expected_generations[index] ||
    row.sequence_token_start() != sequence_start ||
    row.token_start() != token_start ||
    row.token_length() != token_length ||
    row.page_start() != index * 2 ||
    row.page_length() != 2 ||
    row.capability_start() != index * FIXTURE_CAPABILITIES_PER_ROW ||
    row.capability_length() != FIXTURE_CAPABILITIES_PER_ROW ||
    row.produces_token() != (index != 0) ||
    row.completion_slot() != cycle.expected_slots[index] {
    alloc_probe_abort()
  }
  if index == 0 {
    if row.sampling_mode_value() != 0 {
      alloc_probe_abort()
    }
  } else if row.sampling_mode_value() != (if stochastic { 1 } else { 0 }) ||
    row.sampling_seed_value() != seed ||
    row.sample_index() != sample_index ||
    row.sampling_temperature() != (if stochastic { 0.75 } else { 1.0 }) ||
    row.sampling_has_top_k() != stochastic ||
    row.sampling_has_top_p() != stochastic ||
    (
      stochastic &&
      (row.sampling_top_k_value() != 4 || row.sampling_top_p_value() != 0.9)
    ) {
    alloc_probe_abort()
  }
}

///|
fn authenticate_batch_cycle(cycle : PreparedBatchCycle) -> Unit raise {
  let even = cycle.frame.sequence_value() % 2UL == 0UL
  if cycle.frame.sequence_value() != cycle.plan.sequence_wire() ||
    cycle.frame.model_generation_value() != 1UL ||
    cycle.frame.token_budget() != FIXTURE_MAX_TOKENS ||
    cycle.frame.token_count() != FIXTURE_MAX_TOKENS ||
    cycle.frame.page_count() != FIXTURE_MAX_PAGES ||
    cycle.frame.capability_count() != FIXTURE_MAX_CAPABILITIES ||
    cycle.frame.prefill_count() != 3 ||
    cycle.frame.decode_count() != 1 {
    alloc_probe_abort()
  }
  for index in 0..<FIXTURE_MAX_TOKENS {
    if cycle.frame.token_at(index) != index {
      alloc_probe_abort()
    }
  }
  for index in 0..<FIXTURE_MAX_PAGES {
    if cycle.frame.page_index_at(index) != cycle.pages[index].index() ||
      cycle.frame.page_generation_at(index) != cycle.pages[index].generation() {
      alloc_probe_abort()
    }
  }
  for index in 0..<FIXTURE_MAX_CAPABILITIES {
    if cycle.frame.capability_value_at(index) !=
      cycle.plan.capability_at(index).as_int() {
      alloc_probe_abort()
    }
  }
  authenticate_batch_prefill(cycle, 0, 0, 3, 2, false)
  authenticate_batch_prefill(cycle, 1, 3, 3, 3, even)
  authenticate_batch_prefill(cycle, 2, 6, 1, 7, !even)
  let decode = cycle.frame.decode_row(0)
  let seed = cycle.frame.sequence_value() * 16UL + 103UL
  let sample_index = ((cycle.frame.sequence_value() + 2UL) % 8UL).to_int()
  if decode.request_id_value() != cycle.expected_ids[3] ||
    decode.request_generation_value() != cycle.expected_generations[3] ||
    decode.sequence_token_position() != 7 ||
    decode.token_start() != 7 ||
    decode.page_start() != 6 ||
    decode.page_length() != 2 ||
    decode.capability_start() != 24 ||
    decode.capability_length() != FIXTURE_CAPABILITIES_PER_ROW ||
    decode.completion_slot() != cycle.expected_slots[3] ||
    decode.sampling_mode_value() != (if even { 1 } else { 0 }) ||
    decode.sampling_seed_value() != seed ||
    decode.sample_index() != sample_index ||
    decode.sampling_temperature() != (if even { 0.75 } else { 1.0 }) ||
    decode.sampling_has_top_k() != even ||
    decode.sampling_has_top_p() != even ||
    (
      even &&
      (
        decode.sampling_top_k_value() != 4 ||
        decode.sampling_top_p_value() != 0.9
      )
    ) {
    alloc_probe_abort()
  }
}

///|
fn authenticate_batch_completion(
  cycle : PreparedBatchCycle,
  completion : @worker_wire.ValidatedCompletionFrame,
) -> Unit raise {
  if completion.sequence_value() != cycle.plan.sequence_wire() ||
    completion.model_generation_value() != 1UL ||
    completion.entry_count() != FIXTURE_MAX_ROWS {
    alloc_probe_abort()
  }
  for index in 0..<FIXTURE_MAX_ROWS {
    let entry = completion.entry(index)
    let by_slot = completion.entry_for_slot(cycle.expected_slots[index])
    if entry.slot() != cycle.expected_slots[index] ||
      entry.kind() != cycle.expected_kinds[index] ||
      entry.request_id_value() != cycle.expected_ids[index] ||
      entry.request_generation_value() != cycle.expected_generations[index] ||
      entry.processed_tokens() != cycle.expected_processed[index] ||
      entry.token_id() != cycle.expected_tokens[index] ||
      by_slot.request_id_value() != cycle.expected_ids[index] ||
      by_slot.request_generation_value() != cycle.expected_generations[index] {
      alloc_probe_abort()
    }
  }
}

///|
fn execute_batch_cycle(
  owner : @device_worker.DeviceWorkerOwner,
  completion_owner : @worker_wire.CompletionFrameBuffer,
  cycle : PreparedBatchCycle,
  operation_count : UInt64,
) -> Unit raise {
  authenticate_batch_cycle(cycle)
  let copies = fake_copy_count()
  let launches = fake_launch_count()
  let syncs = fake_sync_count()
  let reads = fake_readback_count()
  let creates = fake_resource_create_count()
  let closes = fake_resource_close_count()
  let writer = completion_owner.writer_for(cycle.frame)
  let completion = owner.execute_frame(cycle.frame, writer)
  completion.validate_for(cycle.plan)
  authenticate_batch_completion(cycle, completion)
  if fake_copy_count() - copies != 7UL ||
    fake_launch_count() - launches != operation_count ||
    fake_sync_count() - syncs != 1UL ||
    fake_readback_count() - reads != 3UL ||
    fake_resource_create_count() != creates ||
    fake_resource_close_count() != closes {
    alloc_probe_abort()
  }
  let retired = cycle.plan_owner.retire(cycle.plan)
  cycle.plan_owner.reset(retired)
}

///|
fn authenticate_completion_failure_surface(
  completion : @worker_wire.ValidatedCompletionFrame,
) -> Unit raise {
  for index in 0..<completion.entry_count() {
    try completion.entry(index).failure() catch {
      InvalidFrame(CompletionEntry) => ()
      _ => alloc_probe_abort()
    } noraise {
      _ => alloc_probe_abort()
    }
  }
}
