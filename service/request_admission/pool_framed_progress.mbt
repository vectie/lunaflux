///|
fn LunaRequestPreparationPool::progress_framed_waiting(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  let state = lane.require_framed_work().state() catch {
      _ => raise Invalid(FrameContract)
    }
  match state {
    LunaFramedRequestReceivingPrefix | LunaFramedRequestReceivingBody =>
      if self.draining {
        lane.phase = LUNA_PREPARATION_CANCEL_REQUESTED
      }
    LunaFramedRequestValidating => lane.phase = LUNA_PREPARATION_FRAMED_SCANNING
    LunaFramedRequestReady => lane.phase = LUNA_PREPARATION_FRAMED_TAKE_VIEW
    LunaFramedRequestFailed => raise Invalid(FrameContract)
  }
}

///|
fn LunaRequestPreparationPool::progress_framed_scanner(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Int raise RequestAdmissionError {
  ignore(self)
  let work = lane.require_framed_work()
  ignore(work.progress() catch { _ => raise Invalid(FrameContract) })
  let charged = work.last_work_units() catch {
    _ => raise Invalid(FrameContract)
  }
  if charged != 1 {
    raise Invalid(FrameContract)
  }
  let state = work.state() catch { _ => raise Invalid(FrameContract) }
  match state {
    LunaFramedRequestReceivingPrefix | LunaFramedRequestReceivingBody =>
      lane.phase = LUNA_PREPARATION_FRAMED_RECEIVING
    LunaFramedRequestValidating => ()
    LunaFramedRequestReady => lane.phase = LUNA_PREPARATION_FRAMED_TAKE_VIEW
    LunaFramedRequestFailed => raise Invalid(FrameContract)
  }
  charged
}

///|
fn LunaRequestPreparationPool::progress_framed_take_view(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  lane.framed_view.push(
    lane.require_framed_work().take_view() catch {
      _ => raise Invalid(FrameContract)
    },
  )
  lane.framed_work.clear()
  let view_inference = lane.require_framed_view().inference_limits() catch {
      _ => raise Invalid(FrameContract)
    }
  if view_inference != self.inference {
    raise Invalid(RequestContract)
  }
  lane.framed_model_index = 0
  lane.phase = LUNA_PREPARATION_FRAMED_BIND_MODEL
}

///|
fn LunaRequestPreparationPool::progress_framed_bind_model(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  let index = lane.framed_model_index
  if index >= 128 {
    lane.phase = LUNA_PREPARATION_FRAMED_BIND_SCALARS
    return
  }
  let expected = if index < 64 {
    self.expected_model.content().as_hex().unsafe_get(index).to_int().to_byte()
  } else {
    self.expected_model
    .plan()
    .as_hex()
    .unsafe_get(index - 64)
    .to_int()
    .to_byte()
  }
  let actual = if index < 64 {
    lane.require_framed_view().content_digest_byte_at(index) catch {
      _ => raise Invalid(FrameContract)
    }
  } else {
    lane.require_framed_view().plan_digest_byte_at(index - 64) catch {
      _ => raise Invalid(FrameContract)
    }
  }
  if actual != expected {
    raise Invalid(ModelIdentity)
  }
  lane.framed_model_index += 1
}

///|
fn LunaRequestPreparationPool::progress_framed_bind_scalars(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  let view = lane.require_framed_view()
  let max_new_tokens = view.max_new_tokens() catch {
    _ => raise Invalid(FrameContract)
  }
  let context_ceiling = view.context_ceiling() catch {
    _ => raise Invalid(FrameContract)
  }
  let effective = @inference.EffectiveLimits::new(
    max_new_tokens~,
    context_ceiling~,
    self.inference,
  ) catch {
    _ => raise Invalid(Input)
  }
  let input_capacity = Int::min(
    effective.context_ceiling() - effective.max_new_tokens(),
    Int::min(
      self.inference.max_input_tokens(),
      self.tokenizer.limits().max_output_tokens,
    ),
  )
  if input_capacity <= 0 {
    raise Invalid(Input)
  }
  let deadline_millis = view.deadline_millis() catch {
    _ => raise Invalid(FrameContract)
  }
  if lane.trusted_receipt_bound &&
    deadline_millis != lane.trusted_deadline_millis {
    raise Invalid(Deadline)
  }
  let deadline_budget = @inference.DeadlineBudget::from_milliseconds(
    deadline_millis,
    self.inference,
  ) catch {
    _ => raise Invalid(Deadline)
  }
  let deadline = @inference.AdmissionDeadline::from_receipt(
    lane.receipt_at_millis,
    deadline_budget,
  ) catch {
    _ => raise Invalid(Deadline)
  }
  validate_pool_time(lane.receipt_at_millis, deadline, (self.clock_read)())
  lane.deadline = Some(deadline)
  lane.effective = Some(effective)
  lane.input_capacity = input_capacity
  lane.framed_request_id = view.request_id_value() catch {
    _ => raise Invalid(FrameContract)
  }
  lane.framed_input_count = view.input_length() catch {
    _ => raise Invalid(FrameContract)
  }
  let input_kind = view.input_kind() catch { _ => raise Invalid(FrameContract) }
  lane.framed_input_kind = match input_kind {
    LunaFramedRequestText => 1
    LunaFramedRequestTokenIds => 2
  }
  let sampling_mode = view.sampling_mode() catch {
    _ => raise Invalid(FrameContract)
  }
  lane.framed_sampling_mode = match sampling_mode {
    Greedy => 0
    Sample => 1
  }
  lane.framed_temperature = view.sampling_temperature() catch {
    _ => raise Invalid(FrameContract)
  }
  lane.framed_has_top_k = view.has_top_k() catch {
    _ => raise Invalid(FrameContract)
  }
  lane.framed_top_k = view.top_k() catch { _ => raise Invalid(FrameContract) }
  lane.framed_has_top_p = view.has_top_p() catch {
    _ => raise Invalid(FrameContract)
  }
  lane.framed_top_p = view.top_p() catch { _ => raise Invalid(FrameContract) }
  lane.framed_sampling_seed = view.sampling_seed_value() catch {
    _ => raise Invalid(FrameContract)
  }
  let stream_preference = view.stream_preference() catch {
    _ => raise Invalid(FrameContract)
  }
  lane.framed_streaming = match stream_preference {
    Streaming => true
    TokenIdsStreaming => true
    Buffered => false
  }
  lane.framed_token_ids_only = stream_preference == TokenIdsStreaming
  lane.framed_stop_token_count = view.stop_token_count() catch {
    _ => raise Invalid(FrameContract)
  }
  lane.framed_stop_string_count = view.stop_string_count() catch {
    _ => raise Invalid(FrameContract)
  }
  lane.framed_cache_scope_length = view.cache_scope_length() catch {
    _ => raise Invalid(FrameContract)
  }
  lane.copy_index = 0
  lane.phase = LUNA_PREPARATION_FRAMED_INPUT_BEGIN
}

///|
fn LunaRequestPreparationPool::progress_framed_input_begin(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  if lane.framed_input_kind == 1 {
    lane.tokenizer_input_write = Some(
      lane.tokenizer.begin_luna_input(
        byte_count=lane.framed_input_count,
        special_policy=Reject,
        output_limit=lane.input_capacity,
        truncation=RejectOverflow,
      ) catch {
        _ => raise Invalid(Tokenization)
      },
    )
  } else {
    if lane.framed_input_count <= 0 ||
      lane.framed_input_count > lane.input_capacity {
      raise Invalid(Input)
    }
    lane.token_write = Some(
      lane.token_storage.begin(token_count=lane.framed_input_count) catch {
        _ => raise Invalid(Input)
      },
    )
  }
  lane.copy_index = 0
  lane.phase = LUNA_PREPARATION_FRAMED_INPUT_COPY
}

///|
fn LunaRequestPreparationPool::progress_framed_input_copy(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  if lane.copy_index < lane.framed_input_count {
    if lane.framed_input_kind == 1 {
      let write = match lane.tokenizer_input_write {
        Some(value) => value
        None => abort("Luna framed tokenizer write invariant")
      }
      write.push_byte(
        lane.require_framed_view().input_byte_at(lane.copy_index) catch {
          _ => raise Invalid(FrameContract)
        },
      ) catch {
        _ => raise Invalid(Tokenization)
      }
    } else {
      let write = match lane.token_write {
        Some(value) => value
        None => abort("Luna framed token write invariant")
      }
      write.push(
        lane.require_framed_view().input_token_at(lane.copy_index) catch {
          _ => raise Invalid(FrameContract)
        },
      ) catch {
        _ => raise Invalid(Input)
      }
    }
    lane.copy_index += 1
    return
  }
  if lane.framed_input_kind == 1 {
    let write = match lane.tokenizer_input_write {
      Some(value) => value
      None => abort("Luna framed tokenizer finish invariant")
    }
    lane.tokenizer_work = Some(
      write.finish() catch {
        _ => raise Invalid(Tokenization)
      },
    )
    lane.tokenizer_input_write = None
  } else {
    let write = match lane.token_write {
      Some(value) => value
      None => abort("Luna framed token seal invariant")
    }
    let lease = write.seal() catch { _ => raise Invalid(Input) }
    lane.token_write = None
    lane.token_lease = Some(lease)
    lane.tokens = Some(lease.token_buffer() catch { _ => raise Invalid(Input) })
  }
  lane.phase = LUNA_PREPARATION_FRAMED_SEMANTIC_BEGIN
}

///|
fn LunaRequestPreparationPool::progress_framed_semantic_begin(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  let view = lane.require_framed_view()
  lane.semantic_write.push(
    lane.semantic_storage.begin(
      stop_token_count=lane.framed_stop_token_count,
      stop_string_count=lane.framed_stop_string_count,
      cache_scope_byte_count=lane.framed_cache_scope_length,
      cache_permission=view.cache_permission() catch {
        _ => raise Invalid(FrameContract)
      },
    ) catch {
      _ => raise Invalid(RequestContract)
    },
  )
  lane.semantic_item_index = 0
  lane.semantic_unit_offset = 0
  lane.phase = LUNA_PREPARATION_FRAMED_SEMANTIC_TOKEN_COPY
}

///|
fn LunaRequestPreparationPool::progress_framed_semantic_token_copy(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  if lane.semantic_item_index < lane.framed_stop_token_count {
    lane
    .require_semantic_write()
    .push_stop_token(
      lane.require_framed_view().stop_token_at(lane.semantic_item_index) catch {
        _ => raise Invalid(FrameContract)
      },
    ) catch {
      _ => raise Invalid(RequestContract)
    }
    lane.semantic_item_index += 1
    return
  }
  lane.semantic_item_index = 0
  lane.semantic_unit_offset = 0
  lane.phase = LUNA_PREPARATION_FRAMED_SEMANTIC_STRING_HEADER
}

///|
fn LunaRequestPreparationPool::progress_framed_semantic_string_header(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  if lane.semantic_item_index >= lane.framed_stop_string_count {
    lane.semantic_unit_offset = 0
    lane.phase = LUNA_PREPARATION_FRAMED_SEMANTIC_CACHE_COPY
    return
  }
  let length = lane
    .require_framed_view()
    .stop_string_length(lane.semantic_item_index) catch {
      _ => raise Invalid(FrameContract)
    }
  lane.require_semantic_write().push_stop_string_length(length) catch {
    _ => raise Invalid(RequestContract)
  }
  lane.semantic_byte_length = length
  lane.semantic_unit_offset = 0
  lane.phase = LUNA_PREPARATION_FRAMED_SEMANTIC_STRING_COPY
}

///|
fn LunaRequestPreparationPool::progress_framed_semantic_string_copy(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  if lane.semantic_unit_offset < lane.semantic_byte_length {
    lane
    .require_semantic_write()
    .push_stop_string_byte(
      lane
      .require_framed_view()
      .stop_string_byte_at(lane.semantic_item_index, lane.semantic_unit_offset) catch {
        _ => raise Invalid(FrameContract)
      },
    ) catch {
      _ => raise Invalid(RequestContract)
    }
    lane.semantic_unit_offset += 1
    return
  }
  lane.semantic_item_index += 1
  lane.phase = LUNA_PREPARATION_FRAMED_SEMANTIC_STRING_HEADER
}

///|
fn LunaRequestPreparationPool::progress_framed_semantic_cache_copy(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  if lane.semantic_unit_offset < lane.framed_cache_scope_length {
    lane
    .require_semantic_write()
    .push_cache_scope_byte(
      lane.require_framed_view().cache_scope_byte_at(lane.semantic_unit_offset) catch {
        _ => raise Invalid(FrameContract)
      },
    ) catch {
      _ => raise Invalid(RequestContract)
    }
    lane.semantic_unit_offset += 1
    return
  }
  lane.phase = LUNA_PREPARATION_FRAMED_SEMANTIC_FINISH
}

///|
fn LunaRequestPreparationPool::progress_framed_semantic_finish(
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
  lane.phase = LUNA_PREPARATION_FRAMED_RELEASE_VIEW
}

///|
fn LunaRequestPreparationPool::progress_framed_release_view(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  lane.require_framed_view().release() catch {
    _ => raise Invalid(FrameContract)
  }
  lane.framed_view.clear()
  if lane.framed_input_kind == 1 {
    lane.phase = LUNA_PREPARATION_TEXT_TOKENIZE
  } else {
    lane.phase = LUNA_PREPARATION_SEMANTIC_VALIDATE
  }
}

///|
fn LunaRequestPreparationPool::luna_preparation_pool_progress_disposition(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> LunaRequestPreparationPoolProgress {
  if !self.draining &&
    lane.framed_direct &&
    !lane.framed_receipt_complete &&
    lane.phase == LUNA_PREPARATION_FRAMED_RECEIVING {
    LunaRequestPreparationPoolAwaitingInput
  } else {
    LunaRequestPreparationPoolAdvanced
  }
}
