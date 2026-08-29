///|
pub fn LunaRequestPreparationPool::try_begin_luna_text_handoff_with_receipt(
  self : LunaRequestPreparationPool,
  receipt : LunaRequestReceipt,
  source : @inference.LunaTextRequestHandoffLease,
) -> LunaRequestPreparationAdmission raise RequestAdmissionError {
  let receipt_workspace = receipt.require_workspace()
  if self.draining {
    return preparation_admission(
      self,
      LunaRequestPreparationDraining,
      -1,
      0UL,
      0,
    )
  }
  if self.free_count == 0 {
    return preparation_admission(
      self,
      LunaRequestPreparationSaturated,
      -1,
      0UL,
      0,
    )
  }
  let next_lane = self.free_ring[self.free_head]
  if self.lanes[next_lane].generation == 0xffffffffffffffffUL {
    self.draining = true
    return preparation_admission(
      self,
      LunaRequestPreparationDraining,
      -1,
      0UL,
      0,
    )
  }
  let source_limits = source.inference_limits() catch {
    _ => raise Invalid(RequestContract)
  }
  let source_model = source.model_identity() catch {
    _ => raise Invalid(RequestContract)
  }
  let source_deadline = source.deadline() catch { _ => raise Invalid(Deadline) }
  let prompt_length = source.prompt_length() catch { _ => raise Invalid(Input) }
  let effective = source.effective_limits() catch { _ => raise Invalid(Input) }
  let request_id = source.request_id() catch {
    _ => raise Invalid(RequestContract)
  }
  let sampling = source.validated_sampling() catch {
    _ => raise Invalid(RequestContract)
  }
  let sampling_seed = source.sampling_seed() catch {
    _ => raise Invalid(RequestContract)
  }
  let stream_preference = source.stream_preference() catch {
    _ => raise Invalid(RequestContract)
  }
  if source_limits != self.inference || source_model != self.expected_model {
    raise Invalid(ModelIdentity)
  }
  if source_deadline.milliseconds() != receipt_workspace.budget_millis ||
    source_deadline.milliseconds() == 0UL ||
    source_deadline.milliseconds() > self.inference.max_deadline_millis() {
    raise Invalid(Deadline)
  }
  let input_capacity = Int::min(
    effective.context_ceiling() - effective.max_new_tokens(),
    Int::min(
      self.inference.max_input_tokens(),
      self.tokenizer.limits().max_output_tokens,
    ),
  )
  if prompt_length <= 0 ||
    prompt_length > self.inference.max_text_bytes() ||
    input_capacity <= 0 {
    raise Invalid(Input)
  }
  ignore(receipt.observe())
  let received_at_millis = receipt_workspace.received_at_millis
  let deadline_at_millis = receipt_workspace.deadline_at_millis
  let budget_millis = receipt_workspace.budget_millis
  let claim = source.take_claim() catch {
    _ => raise Invalid(LunaPreparedLifecycle)
  }
  let semantic = claim.take_semantics() catch {
    _ => {
      claim.release() catch {
        _ => ()
      }
      raise Invalid(RequestContract)
    }
  }
  let lane_index = self.pop_free_lane()
  let lane = self.lanes[lane_index]
  receipt.consume() catch {
    _ => {
      semantic.release() catch {
        _ => ()
      }
      claim.release() catch {
        _ => ()
      }
      abort("Luna typed receipt changed during atomic transfer")
    }
  }
  lane.generation += 1UL
  lane.receipt_at_millis = received_at_millis
  lane.hard_deadline = Some(
    @inference.AdmissionDeadline::from_receipt(
      received_at_millis, source_deadline,
    ) catch {
      _ => abort("Luna typed handoff deadline invariant")
    },
  )
  if lane.hard_deadline.unwrap().at_millis() != deadline_at_millis {
    abort("Luna typed handoff absolute-deadline invariant")
  }
  lane.deadline = lane.hard_deadline
  lane.request = None
  lane.effective = Some(effective)
  lane.input_capacity = input_capacity
  lane.framed_direct = true
  lane.typed_direct = true
  lane.framed_receipt_complete = true
  lane.trusted_receipt_bound = true
  lane.trusted_deadline_millis = budget_millis
  lane.framed_input_kind = 1
  lane.framed_input_count = prompt_length
  lane.typed_request_id = request_id
  lane.typed_sampling = sampling
  lane.typed_sampling_seed = sampling_seed
  lane.typed_stream_preference = stream_preference
  lane.typed_handoff.push(claim)
  lane.semantic_lease.push(semantic)
  lane.copy_index = 0
  lane.phase = LUNA_PREPARATION_TYPED_INPUT_BEGIN
  lane.failure = None
  lane.claim = None
  lane.last_work_units = 0
  lane.total_work_units = 0UL
  self.enqueue_active(lane_index)
  preparation_admission(
    self,
    LunaRequestPreparationAdmitted,
    lane_index,
    lane.generation,
    1,
  )
}

///|
fn LunaRequestPreparationPool::progress_typed_input_begin(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  if lane.typed_handoff.length() != 1 || lane.framed_input_count <= 0 {
    raise Invalid(LunaPreparedLifecycle)
  }
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
  lane.copy_index = 0
  lane.phase = LUNA_PREPARATION_TYPED_INPUT_COPY
}

///|
fn LunaRequestPreparationPool::progress_typed_input_copy(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  let write = match lane.tokenizer_input_write {
    Some(value) => value
    None => abort("Luna typed tokenizer write invariant")
  }
  let source = if lane.typed_handoff.length() == 1 {
    lane.typed_handoff[0]
  } else {
    abort("Luna typed handoff claim invariant")
  }
  if lane.copy_index < lane.framed_input_count {
    write.push_byte(
      source.prompt_byte_at(lane.copy_index) catch {
        _ => raise Invalid(Input)
      },
    ) catch {
      _ => raise Invalid(Tokenization)
    }
    lane.copy_index += 1
    return
  }
  lane.tokenizer_work = Some(
    write.finish() catch {
      _ => raise Invalid(Tokenization)
    },
  )
  lane.tokenizer_input_write = None
  source.release() catch {
    _ => raise Invalid(LunaPreparedLifecycle)
  }
  lane.typed_handoff.clear()
  lane.copy_index = 0
  lane.phase = LUNA_PREPARATION_TEXT_TOKENIZE
}
