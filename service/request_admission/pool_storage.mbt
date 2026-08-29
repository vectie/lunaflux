///|
fn checked_add_cells(
  left : UInt64,
  right : UInt64,
) -> UInt64 raise RequestAdmissionError {
  if left > 0xffffffffffffffffUL - right {
    raise Invalid(LunaPreparationStorage)
  }
  left + right
}

///|
fn checked_multiply_cells(
  left : UInt64,
  right : UInt64,
) -> UInt64 raise RequestAdmissionError {
  if right != 0UL && left > 0xffffffffffffffffUL / right {
    raise Invalid(LunaPreparationStorage)
  }
  left * right
}

///|
fn validate_pool_storage(
  tokenizer : @tokenizer.TokenizerSpec,
  inference : @inference.InferenceLimits,
  framed_limits : @framed_wire.FramedWireLimits,
  lane_count : Int,
  budget : LunaRequestPreparationStorageBudget,
) -> (Int, Int) raise RequestAdmissionError {
  if lane_count <= 0 || lane_count > LUNA_PREPARATION_MAX_LANES {
    raise Invalid(LunaPreparationPool)
  }
  let input_capacity = Int::min(
    tokenizer.limits().max_input_bytes,
    inference.max_text_bytes(),
  )
  let output_capacity = Int::min(
    tokenizer.limits().max_output_tokens,
    inference.max_input_tokens(),
  )
  let tokenizer_int_cells = tokenizer.luna_worker_required_int_cells(
    input_capacity,
  ) catch {
    _ => raise Invalid(LunaPreparationStorage)
  }
  let tokenizer_byte_cells = @tokenizer.LunaTokenizerWorker::required_byte_cells(
    input_capacity,
  ) catch {
    _ => raise Invalid(LunaPreparationStorage)
  }
  let token_cells = @inference.LunaTokenBufferStorage::required_int_cells(
    output_capacity,
  ) catch {
    _ => raise Invalid(LunaPreparationStorage)
  }
  let semantic_int_cells = @inference.LunaRequestSemanticStorage::required_int_cells(
    inference,
  ) catch {
    _ => raise Invalid(LunaPreparationStorage)
  }
  let output_int_cells = (@incremental_output.LunaIncrementalOutputWorkspace::required_int_cells(
    inference,
  ) catch {
    _ => raise Invalid(LunaPreparationStorage)
  }).to_uint64()
  let framed_int_cells = @framed_wire.LunaFramedRequestWorkspace::required_int_cells(
    framed_limits,
  ) catch {
    _ => raise Invalid(LunaPreparationStorage)
  }
  let per_lane_int = checked_add_cells(
    checked_add_cells(
      checked_add_cells(tokenizer_int_cells, token_cells),
      semantic_int_cells,
    ),
    checked_add_cells(output_int_cells, framed_int_cells),
  )
  let output_byte_cells = (@incremental_output.LunaIncrementalOutputWorkspace::required_byte_cells(
    inference,
  ) catch {
    _ => raise Invalid(LunaPreparationStorage)
  }).to_uint64()
  let semantic_byte_cells = @inference.LunaRequestSemanticStorage::required_byte_cells(
    inference,
  ) catch {
    _ => raise Invalid(LunaPreparationStorage)
  }
  let framed_byte_cells = @framed_wire.LunaFramedRequestWorkspace::required_byte_cells(
    framed_limits,
  ) catch {
    _ => raise Invalid(LunaPreparationStorage)
  }
  let per_lane_byte = checked_add_cells(
    checked_add_cells(
      checked_add_cells(tokenizer_byte_cells, output_byte_cells),
      semantic_byte_cells,
    ),
    framed_byte_cells,
  )
  let output_reference_cells = (@incremental_output.LunaIncrementalOutputWorkspace::required_reference_cells(
    inference,
  ) catch {
    _ => raise Invalid(LunaPreparationStorage)
  }).to_uint64()
  let lanes = lane_count.to_uint64()
  let total_int = checked_add_cells(
    checked_multiply_cells(per_lane_int, lanes),
    checked_multiply_cells(lanes, 2UL),
  )
  let total_byte = checked_multiply_cells(per_lane_byte, lanes)
  let total_references = checked_multiply_cells(
    checked_add_cells(7UL, output_reference_cells),
    lanes,
  )
  if total_int > budget.int_cells ||
    total_byte > budget.byte_cells ||
    total_references > budget.reference_cells {
    raise Invalid(LunaPreparationStorage)
  }
  (input_capacity, output_capacity)
}

///|
/// Validate the exact preparation-pool storage and lane envelope without
/// allocating a live pool owner.
pub fn LunaRequestPreparationStorageBudget::require_fits(
  self : LunaRequestPreparationStorageBudget,
  tokenizer : @tokenizer.TokenizerSpec,
  inference : @inference.InferenceLimits,
  framed_limits : @framed_wire.FramedWireLimits,
  lane_count : Int,
) -> Unit raise RequestAdmissionError {
  ignore(
    validate_pool_storage(tokenizer, inference, framed_limits, lane_count, self),
  )
}
