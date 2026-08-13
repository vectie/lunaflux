///|
fn write_sampling_parameters(
  bytes : FixedArray[Byte],
  offset : Int,
  sampling : @inference.SamplingParameters,
) -> Unit {
  write_u32(
    bytes,
    offset,
    match sampling.mode() {
      Greedy => 0
      Sample => 1
    },
  )
  write_u64(bytes, offset + 4, sampling.temperature().reinterpret_as_uint64())
  match sampling.top_k() {
    Some(value) => {
      write_u32(bytes, offset + 12, 1)
      write_i32(bytes, offset + 16, value)
    }
    None => {
      write_u32(bytes, offset + 12, 0)
      write_i32(bytes, offset + 16, 0)
    }
  }
  match sampling.top_p() {
    Some(value) => {
      write_u32(bytes, offset + 20, 1)
      write_u64(bytes, offset + 24, value.reinterpret_as_uint64())
    }
    None => {
      write_u32(bytes, offset + 20, 0)
      write_u64(bytes, offset + 24, 0UL)
    }
  }
}

///|
fn validate_plan_counts_for_encode(
  owner : PlanFrameBuffer,
  token_budget : Int,
  token_count : Int,
  page_count : Int,
  capability_count : Int,
  prefill_count : Int,
  decode_count : Int,
) -> Unit raise WorkerWireError {
  let limits = owner.worker_limits
  if token_budget < token_count ||
    token_budget > limits.max_plan_tokens() ||
    token_count < 0 ||
    token_count > limits.max_plan_tokens() ||
    page_count < 0 ||
    page_count > limits.max_plan_pages() ||
    capability_count < 0 ||
    capability_count > limits.max_capabilities() ||
    prefill_count < 0 ||
    prefill_count > limits.max_prefill_rows() ||
    decode_count < 0 ||
    decode_count > limits.max_decode_rows() ||
    prefill_count + decode_count <= 0 ||
    prefill_count + decode_count > limits.max_plan_rows() {
    raise InvalidFrame(Counts)
  }
}

///|
fn validate_sampling_for_encode(
  sampling : @inference.SamplingParameters,
  limits : @inference.InferenceLimits,
) -> Unit raise WorkerWireError {
  match sampling.mode() {
    Greedy =>
      if sampling.temperature() != 1.0 ||
        sampling.top_k() is Some(_) ||
        sampling.top_p() is Some(_) {
        raise InvalidFrame(Sampling)
      }
    Sample => {
      let temperature = sampling.temperature()
      if temperature.is_nan() ||
        temperature.is_inf() ||
        temperature <= 0.0 ||
        temperature > limits.max_temperature() {
        raise InvalidFrame(Sampling)
      }
      match sampling.top_k() {
        Some(value) if value <= 0 || value > limits.max_top_k() =>
          raise InvalidFrame(Sampling)
        _ => ()
      }
      match sampling.top_p() {
        Some(value) if value.is_nan() ||
          value.is_inf() ||
          value <= 0.0 ||
          value > 1.0 => raise InvalidFrame(Sampling)
        _ => ()
      }
    }
  }
}

///|
fn preflight_plan_for_encode(
  owner : PlanFrameBuffer,
  plan : @worker_protocol.SubmittedSchedulePlan,
  token_count : Int,
  page_count : Int,
  capability_count : Int,
  prefill_count : Int,
  decode_count : Int,
) -> Unit raise WorkerWireError {
  let max_token_id = if owner.worker_limits.max_token_id() <
    owner.inference_limits.max_token_id() {
    owner.worker_limits.max_token_id()
  } else {
    owner.inference_limits.max_token_id()
  }
  for index in 0..<token_count {
    let token = plan.token_at(index) catch { error => raise Protocol(error) }
    if token < 0 || token > max_token_id {
      raise InvalidFrame(TokenTable)
    }
  }
  for index in 0..<page_count {
    let page = plan.page_at(index) catch { error => raise Protocol(error) }
    if page.index() < 0 || page.generation() <= 0 {
      raise InvalidFrame(PageTable)
    }
  }
  for index in 0..<capability_count {
    let capability = try plan.capability_at(index) catch {
      error => raise Protocol(error)
    } noraise {
      value => value
    }
    if capability.as_int() <= 0 {
      raise InvalidFrame(CapabilityTable)
    }
  }
  for index in 0..<prefill_count {
    let row = plan.prefill_row(index) catch { error => raise Protocol(error) }
    ignore(row.request_id() catch { error => raise Protocol(error) })
    ignore(row.request_generation() catch { error => raise Protocol(error) })
    ignore(row.sequence_token_start() catch { error => raise Protocol(error) })
    ignore(row.tokens() catch { error => raise Protocol(error) })
    ignore(row.page_table() catch { error => raise Protocol(error) })
    ignore(row.capabilities() catch { error => raise Protocol(error) })
    ignore(row.produces_token() catch { error => raise Protocol(error) })
    validate_sampling_for_encode(
      row.sampling() catch {
        error => raise Protocol(error)
      },
      owner.inference_limits,
    )
    ignore(row.sampling_seed_value() catch { error => raise Protocol(error) })
    ignore(row.sample_index() catch { error => raise Protocol(error) })
    ignore(row.completion_slot() catch { error => raise Protocol(error) })
  }
  for index in 0..<decode_count {
    let row = plan.decode_row(index) catch { error => raise Protocol(error) }
    ignore(row.request_id() catch { error => raise Protocol(error) })
    ignore(row.request_generation() catch { error => raise Protocol(error) })
    ignore(
      row.sequence_token_position() catch {
        error => raise Protocol(error)
      },
    )
    ignore(row.input_token() catch { error => raise Protocol(error) })
    ignore(row.page_table() catch { error => raise Protocol(error) })
    ignore(row.capabilities() catch { error => raise Protocol(error) })
    validate_sampling_for_encode(
      row.sampling() catch {
        error => raise Protocol(error)
      },
      owner.inference_limits,
    )
    ignore(row.sampling_seed_value() catch { error => raise Protocol(error) })
    ignore(row.sample_index() catch { error => raise Protocol(error) })
    ignore(row.completion_slot() catch { error => raise Protocol(error) })
  }
}

///|
fn write_plan_header(
  bytes : FixedArray[Byte],
  length : Int,
  sequence : UInt64,
  generation : UInt64,
  token_budget : Int,
  token_count : Int,
  page_count : Int,
  capability_count : Int,
  prefill_count : Int,
  decode_count : Int,
) -> Unit {
  write_u32(bytes, 0, plan_magic)
  write_u32(bytes, 4, wire_version)
  write_u32(bytes, 8, plan_kind)
  write_i32(bytes, 12, length)
  write_u64(bytes, 16, sequence)
  write_u64(bytes, 24, generation)
  write_i32(bytes, 32, token_budget)
  write_i32(bytes, 36, token_count)
  write_i32(bytes, 40, page_count)
  write_i32(bytes, 44, capability_count)
  write_i32(bytes, 48, prefill_count)
  write_i32(bytes, 52, decode_count)
  write_u32(bytes, 56, 0)
  write_u32(bytes, plan_checksum_offset, 0)
}

///|
fn write_prefill_row(
  bytes : FixedArray[Byte],
  offset : Int,
  row : @worker_protocol.SubmittedPrefillRow,
) -> Unit raise WorkerWireError {
  let request_id = row.request_id() catch { error => raise Protocol(error) }
  let generation = row.request_generation() catch {
    error => raise Protocol(error)
  }
  let tokens = row.tokens() catch { error => raise Protocol(error) }
  let pages = row.page_table() catch { error => raise Protocol(error) }
  let capabilities = row.capabilities() catch { error => raise Protocol(error) }
  let sampling = row.sampling() catch { error => raise Protocol(error) }
  let completion = row.completion_slot() catch {
    error => raise Protocol(error)
  }
  let produces_token = try row.produces_token() catch {
    error => raise Protocol(error)
  } noraise {
    value => value
  }
  write_u64(bytes, offset, request_id.value())
  write_u64(bytes, offset + 8, generation.value())
  write_i32(
    bytes,
    offset + 16,
    row.sequence_token_start() catch {
      error => raise Protocol(error)
    },
  )
  write_i32(bytes, offset + 20, tokens.start())
  write_i32(bytes, offset + 24, tokens.length())
  write_i32(bytes, offset + 28, pages.start())
  write_i32(bytes, offset + 32, pages.length())
  write_i32(bytes, offset + 36, capabilities.start())
  write_i32(bytes, offset + 40, capabilities.length())
  write_u32(bytes, offset + 44, if produces_token { 1 } else { 0 })
  write_sampling_parameters(bytes, offset + 48, sampling)
  write_u64(
    bytes,
    offset + 80,
    row.sampling_seed_value() catch {
      error => raise Protocol(error)
    },
  )
  write_i32(
    bytes,
    offset + 88,
    row.sample_index() catch {
      error => raise Protocol(error)
    },
  )
  write_i32(bytes, offset + 92, completion.value())
}

///|
fn write_decode_row(
  bytes : FixedArray[Byte],
  offset : Int,
  row : @worker_protocol.SubmittedDecodeRow,
) -> Unit raise WorkerWireError {
  let request_id = row.request_id() catch { error => raise Protocol(error) }
  let generation = row.request_generation() catch {
    error => raise Protocol(error)
  }
  let token = row.input_token() catch { error => raise Protocol(error) }
  let pages = row.page_table() catch { error => raise Protocol(error) }
  let capabilities = row.capabilities() catch { error => raise Protocol(error) }
  let sampling = row.sampling() catch { error => raise Protocol(error) }
  let completion = row.completion_slot() catch {
    error => raise Protocol(error)
  }
  write_u64(bytes, offset, request_id.value())
  write_u64(bytes, offset + 8, generation.value())
  write_i32(
    bytes,
    offset + 16,
    row.sequence_token_position() catch {
      error => raise Protocol(error)
    },
  )
  write_i32(bytes, offset + 20, token.start())
  write_i32(bytes, offset + 24, pages.start())
  write_i32(bytes, offset + 28, pages.length())
  write_i32(bytes, offset + 32, capabilities.start())
  write_i32(bytes, offset + 36, capabilities.length())
  write_sampling_parameters(bytes, offset + 40, sampling)
  write_u64(
    bytes,
    offset + 72,
    row.sampling_seed_value() catch {
      error => raise Protocol(error)
    },
  )
  write_i32(
    bytes,
    offset + 80,
    row.sample_index() catch {
      error => raise Protocol(error)
    },
  )
  write_i32(bytes, offset + 84, completion.value())
}

///|
/// Encodes one exact submitted plan into startup-owned fixed storage.
/// Successful calls grow no collection and invalidate every older frame view.
pub fn PlanFrameBuffer::encode_plan(
  self : PlanFrameBuffer,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> ValidatedPlanFrame raise WorkerWireError {
  let sequence = plan.sequence_wire() catch { error => raise Protocol(error) }
  let generation = plan.model_plan_generation_wire() catch {
    error => raise Protocol(error)
  }
  let token_budget = plan.token_budget() catch {
    error => raise Protocol(error)
  }
  let token_count = plan.token_count() catch { error => raise Protocol(error) }
  let page_count = plan.page_count() catch { error => raise Protocol(error) }
  let capability_count = plan.capability_count() catch {
    error => raise Protocol(error)
  }
  let prefill_count = plan.prefill_count() catch {
    error => raise Protocol(error)
  }
  let decode_count = plan.decode_count() catch {
    error => raise Protocol(error)
  }
  validate_plan_counts_for_encode(
    self, token_budget, token_count, page_count, capability_count, prefill_count,
    decode_count,
  )
  let length = exact_plan_frame_bytes(
    token_count, page_count, capability_count, prefill_count, decode_count,
  )
  if length > self.storage.length() {
    raise InvalidLimit(PlanFrameBytes)
  }
  if self.epoch == 0xffffffffffffffffUL {
    raise FrameEpochOverflow
  }
  preflight_plan_for_encode(
    self, plan, token_count, page_count, capability_count, prefill_count, decode_count,
  )
  write_plan_header(
    self.storage,
    length,
    sequence,
    generation,
    token_budget,
    token_count,
    page_count,
    capability_count,
    prefill_count,
    decode_count,
  )
  let token_offset = token_table_offset()
  for index in 0..<token_count {
    write_i32(
      self.storage,
      token_offset + index * 4,
      plan.token_at(index) catch {
        error => raise Protocol(error)
      },
    )
  }
  let page_offset = page_table_offset(token_count)
  for index in 0..<page_count {
    let page = plan.page_at(index) catch { error => raise Protocol(error) }
    write_i32(self.storage, page_offset + index * 8, page.index())
    write_i32(self.storage, page_offset + index * 8 + 4, page.generation())
  }
  let capability_offset = capability_table_offset(token_count, page_count)
  for index in 0..<capability_count {
    let capability = try plan.capability_at(index) catch {
      error => raise Protocol(error)
    } noraise {
      value => value
    }
    write_i32(self.storage, capability_offset + index * 4, capability.as_int())
  }
  let prefill_offset = prefill_table_offset(
    token_count, page_count, capability_count,
  )
  for index in 0..<prefill_count {
    write_prefill_row(
      self.storage,
      prefill_offset + index * prefill_row_bytes,
      plan.prefill_row(index) catch {
        error => raise Protocol(error)
      },
    )
  }
  let decode_offset = decode_table_offset(
    token_count, page_count, capability_count, prefill_count,
  )
  for index in 0..<decode_count {
    write_decode_row(
      self.storage,
      decode_offset + index * decode_row_bytes,
      plan.decode_row(index) catch {
        error => raise Protocol(error)
      },
    )
  }
  write_u32(
    self.storage,
    plan_checksum_offset,
    frame_checksum(self.storage, length),
  )
  self.epoch += 1UL
  self.length = length
  ValidatedPlanFrame(self, self.epoch)
}
