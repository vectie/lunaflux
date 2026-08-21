///|
fn LunaRequestPreparationPool::progress_semantic_cache_measure(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit {
  ignore(self)
  let scope = lane.require_request().cache().scope().as_string()
  if lane.semantic_unit_offset < scope.length() {
    ignore(scope.unsafe_get(lane.semantic_unit_offset))
    lane.semantic_unit_offset += 1
    lane.semantic_byte_length += 1
    return
  }
  lane.semantic_unit_offset = 0
  lane.phase = LUNA_PREPARATION_SEMANTIC_BEGIN
}

///|
fn LunaRequestPreparationPool::progress_semantic_begin(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  let request = lane.require_request()
  lane.semantic_write.push(
    lane.semantic_storage.begin(
      stop_token_count=request.stops().token_ids().length(),
      stop_string_count=request.stops().strings().length(),
      cache_scope_byte_count=lane.semantic_byte_length,
      cache_permission=request.cache().permission(),
    ) catch {
      _ => raise Invalid(RequestContract)
    },
  )
  lane.semantic_item_index = 0
  lane.semantic_byte_length = 0
  lane.phase = LUNA_PREPARATION_SEMANTIC_TOKEN_COPY
}

///|
fn LunaRequestPreparationPool::progress_semantic_token_copy(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  let tokens = lane.require_request().stops().token_ids()
  if lane.semantic_item_index < tokens.length() {
    lane
    .require_semantic_write()
    .push_stop_token(tokens[lane.semantic_item_index]) catch {
      _ => raise Invalid(RequestContract)
    }
    lane.semantic_item_index += 1
    return
  }
  lane.semantic_item_index = 0
  lane.semantic_unit_offset = 0
  lane.semantic_byte_length = 0
  lane.phase = LUNA_PREPARATION_SEMANTIC_STRING_MEASURE
}

///|
fn LunaRequestPreparationPool::progress_semantic_string_measure(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit {
  ignore(self)
  let strings = lane.require_request().stops().strings()
  if lane.semantic_item_index >= strings.length() {
    lane.semantic_unit_offset = 0
    lane.phase = LUNA_PREPARATION_SEMANTIC_CACHE_EMIT
    return
  }
  let value = strings[lane.semantic_item_index]
  if lane.semantic_unit_offset >= value.length() {
    lane.semantic_unit_offset = 0
    lane.phase = LUNA_PREPARATION_SEMANTIC_STRING_HEADER
    return
  }
  let codepoint = luna_semantic_codepoint_at(value, lane.semantic_unit_offset)
  lane.semantic_unit_offset += luna_semantic_codepoint_units_at(
    value,
    lane.semantic_unit_offset,
  )
  lane.semantic_byte_length += luna_semantic_utf8_width(codepoint)
}

///|
fn LunaRequestPreparationPool::progress_semantic_string_header(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  lane
  .require_semantic_write()
  .push_stop_string_length(lane.semantic_byte_length) catch {
    _ => raise Invalid(RequestContract)
  }
  lane.semantic_unit_offset = 0
  lane.semantic_codepoint_width = 0
  lane.semantic_codepoint_byte_index = 0
  lane.phase = LUNA_PREPARATION_SEMANTIC_STRING_EMIT
}

///|
fn LunaRequestPreparationPool::progress_semantic_string_emit(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  let strings = lane.require_request().stops().strings()
  let value = strings[lane.semantic_item_index]
  if lane.semantic_codepoint_byte_index >= lane.semantic_codepoint_width {
    if lane.semantic_unit_offset >= value.length() {
      lane.semantic_item_index += 1
      lane.semantic_unit_offset = 0
      lane.semantic_byte_length = 0
      lane.semantic_codepoint_width = 0
      lane.semantic_codepoint_byte_index = 0
      lane.phase = LUNA_PREPARATION_SEMANTIC_STRING_MEASURE
      return
    }
    let codepoint = luna_semantic_codepoint_at(value, lane.semantic_unit_offset)
    lane.semantic_codepoint = codepoint
    lane.semantic_codepoint_width = luna_semantic_utf8_width(codepoint)
    lane.semantic_codepoint_byte_index = 0
    lane.semantic_unit_offset += luna_semantic_codepoint_units_at(
      value,
      lane.semantic_unit_offset,
    )
  }
  lane
  .require_semantic_write()
  .push_stop_string_byte(
    luna_semantic_utf8_byte(
      lane.semantic_codepoint,
      lane.semantic_codepoint_width,
      lane.semantic_codepoint_byte_index,
    ),
  ) catch {
    _ => raise Invalid(RequestContract)
  }
  lane.semantic_codepoint_byte_index += 1
}

///|
fn LunaRequestPreparationPool::progress_semantic_cache_emit(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  let scope = lane.require_request().cache().scope().as_string()
  if lane.semantic_unit_offset < scope.length() {
    let code = scope.unsafe_get(lane.semantic_unit_offset).to_int()
    if code < 0 || code > 0x7f {
      raise Invalid(RequestContract)
    }
    lane.require_semantic_write().push_cache_scope_byte(code.to_byte()) catch {
      _ => raise Invalid(RequestContract)
    }
    lane.semantic_unit_offset += 1
    return
  }
  lane.phase = LUNA_PREPARATION_SEMANTIC_FINISH
}

///|
fn LunaRequestPreparationPool::progress_semantic_finish(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  lane.semantic_work.push(
    lane.require_semantic_write().finish() catch {
      _ => raise Invalid(RequestContract)
    },
  )
  lane.semantic_write.clear()
  lane.phase = LUNA_PREPARATION_SEMANTIC_VALIDATE
}

///|
fn LunaRequestPreparationPool::progress_semantic_validate(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  let progress = lane.require_semantic_work().progress() catch {
      _ => raise Invalid(RequestContract)
    }
  if progress.is_failed() {
    raise Invalid(RequestContract)
  }
  if progress.is_ready() {
    lane.phase = LUNA_PREPARATION_SEMANTIC_TAKE
  }
}

///|
fn LunaRequestPreparationPool::progress_semantic_take(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  let lease = lane.require_semantic_work().take_lease() catch {
      _ => raise Invalid(RequestContract)
    }
  lane.semantic_work.clear()
  lane.semantic_lease.push(lease)
  lane.output_work = Some(
    lane.output_workspace.begin_luna_request_semantics(
      lease.view() catch {
        _ => raise Invalid(RequestContract)
      },
      special_policy=Skip,
    ) catch {
      _ => raise Invalid(OutputContract)
    },
  )
  lane.phase = LUNA_PREPARATION_OUTPUT_SETUP
}
