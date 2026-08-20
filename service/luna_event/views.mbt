///|
fn luna_authenticate_view(
  owner : LunaEventOwner,
  epoch : UInt64,
  kind : LunaEventSlotKind,
) -> LunaEventOwner raise LunaEventError {
  if !owner.busy || owner.epoch != epoch {
    raise luna_event_invalid(LunaEventEpoch)
  }
  if owner.kind != kind {
    raise luna_event_invalid(LunaEventKind)
  }
  owner
}

///|
fn luna_copy_payload(
  owner : LunaEventOwner,
  destination : FixedArray[Byte],
  destination_offset : Int,
) -> Int raise LunaEventError {
  let length = owner.payload_length
  if destination_offset < 0 ||
    destination_offset > destination.length() - length {
    raise luna_event_invalid(LunaEventDestination)
  }
  for index in 0..<length {
    destination[destination_offset + index] = owner.payload[index]
  }
  length
}

///|
pub fn LunaEventView::kind(
  self : LunaEventView,
) -> LunaEventKind raise LunaEventError {
  let owner = self.owner
  if !owner.busy || owner.epoch != self.epoch {
    raise luna_event_invalid(LunaEventEpoch)
  }
  match owner.kind {
    LunaAcceptedEventSlot => LunaAcceptedEventKind
    LunaTokenEventSlot => LunaTokenEventKind
    LunaUsageEventSlot => LunaUsageEventKind
    LunaCompletedEventSlot => LunaCompletedEventKind
    LunaFailedEventSlot => LunaFailedEventKind
    LunaNoEventSlot => abort("luna event view kind invariant")
  }
}

///|
pub fn LunaEventView::accepted(
  self : LunaEventView,
) -> LunaAcceptedEventView raise LunaEventError {
  ignore(luna_authenticate_view(self.owner, self.epoch, LunaAcceptedEventSlot))
  { owner: self.owner, epoch: self.epoch }
}

///|
pub fn LunaEventView::token(
  self : LunaEventView,
) -> LunaTokenEventView raise LunaEventError {
  ignore(luna_authenticate_view(self.owner, self.epoch, LunaTokenEventSlot))
  { owner: self.owner, epoch: self.epoch }
}

///|
pub fn LunaEventView::usage(
  self : LunaEventView,
) -> LunaUsageEventView raise LunaEventError {
  ignore(luna_authenticate_view(self.owner, self.epoch, LunaUsageEventSlot))
  { owner: self.owner, epoch: self.epoch }
}

///|
pub fn LunaEventView::completed(
  self : LunaEventView,
) -> LunaCompletedEventView raise LunaEventError {
  ignore(luna_authenticate_view(self.owner, self.epoch, LunaCompletedEventSlot))
  { owner: self.owner, epoch: self.epoch }
}

///|
pub fn LunaEventView::failed(
  self : LunaEventView,
) -> LunaFailedEventView raise LunaEventError {
  ignore(luna_authenticate_view(self.owner, self.epoch, LunaFailedEventSlot))
  { owner: self.owner, epoch: self.epoch }
}

///|
pub fn LunaAcceptedEventView::model_identity(
  self : LunaAcceptedEventView,
) -> @spec.ModelIdentity raise LunaEventError {
  let owner = luna_authenticate_view(
    self.owner,
    self.epoch,
    LunaAcceptedEventSlot,
  )
  match owner.model_identity {
    Some(value) => value
    None => abort("luna accepted event identity invariant")
  }
}

///|
pub fn LunaAcceptedEventView::effective_limits(
  self : LunaAcceptedEventView,
) -> @inference.EffectiveLimits raise LunaEventError {
  let owner = luna_authenticate_view(
    self.owner,
    self.epoch,
    LunaAcceptedEventSlot,
  )
  match owner.effective_limits {
    Some(value) => value
    None => abort("luna accepted event limits invariant")
  }
}

///|
pub fn LunaTokenEventView::token_id(
  self : LunaTokenEventView,
) -> Int raise LunaEventError {
  luna_authenticate_view(self.owner, self.epoch, LunaTokenEventSlot).token_id
}

///|
pub fn LunaTokenEventView::position(
  self : LunaTokenEventView,
) -> Int raise LunaEventError {
  luna_authenticate_view(self.owner, self.epoch, LunaTokenEventSlot).position
}

///|
pub fn LunaTokenEventView::delta_length(
  self : LunaTokenEventView,
) -> Int raise LunaEventError {
  luna_authenticate_view(self.owner, self.epoch, LunaTokenEventSlot).payload_length
}

///|
pub fn LunaTokenEventView::copy_delta_to(
  self : LunaTokenEventView,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> Int raise LunaEventError {
  luna_copy_payload(
    luna_authenticate_view(self.owner, self.epoch, LunaTokenEventSlot),
    destination,
    destination_offset,
  )
}

///|
pub fn LunaUsageEventView::input_tokens(
  self : LunaUsageEventView,
) -> Int raise LunaEventError {
  luna_authenticate_view(self.owner, self.epoch, LunaUsageEventSlot).input_tokens
}

///|
pub fn LunaUsageEventView::cached_input_tokens(
  self : LunaUsageEventView,
) -> Int raise LunaEventError {
  luna_authenticate_view(self.owner, self.epoch, LunaUsageEventSlot).cached_input_tokens
}

///|
pub fn LunaUsageEventView::output_tokens(
  self : LunaUsageEventView,
) -> Int raise LunaEventError {
  luna_authenticate_view(self.owner, self.epoch, LunaUsageEventSlot).output_tokens
}

///|
pub fn LunaUsageEventView::total_tokens(
  self : LunaUsageEventView,
) -> Int raise LunaEventError {
  let owner = luna_authenticate_view(self.owner, self.epoch, LunaUsageEventSlot)
  owner.input_tokens + owner.output_tokens
}

///|
pub fn LunaCompletedEventView::finish_reason(
  self : LunaCompletedEventView,
) -> @inference.FinishReason raise LunaEventError {
  luna_authenticate_view(self.owner, self.epoch, LunaCompletedEventSlot).finish_reason
}

///|
pub fn LunaCompletedEventView::input_tokens(
  self : LunaCompletedEventView,
) -> Int raise LunaEventError {
  luna_authenticate_view(self.owner, self.epoch, LunaCompletedEventSlot).input_tokens
}

///|
pub fn LunaCompletedEventView::cached_input_tokens(
  self : LunaCompletedEventView,
) -> Int raise LunaEventError {
  luna_authenticate_view(self.owner, self.epoch, LunaCompletedEventSlot).cached_input_tokens
}

///|
pub fn LunaCompletedEventView::output_tokens(
  self : LunaCompletedEventView,
) -> Int raise LunaEventError {
  luna_authenticate_view(self.owner, self.epoch, LunaCompletedEventSlot).output_tokens
}

///|
pub fn LunaCompletedEventView::total_tokens(
  self : LunaCompletedEventView,
) -> Int raise LunaEventError {
  let owner = luna_authenticate_view(
    self.owner,
    self.epoch,
    LunaCompletedEventSlot,
  )
  owner.input_tokens + owner.output_tokens
}

///|
pub fn LunaCompletedEventView::final_delta_length(
  self : LunaCompletedEventView,
) -> Int raise LunaEventError {
  luna_authenticate_view(self.owner, self.epoch, LunaCompletedEventSlot).payload_length
}

///|
pub fn LunaCompletedEventView::copy_final_delta_to(
  self : LunaCompletedEventView,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> Int raise LunaEventError {
  luna_copy_payload(
    luna_authenticate_view(self.owner, self.epoch, LunaCompletedEventSlot),
    destination,
    destination_offset,
  )
}

///|
pub fn LunaFailedEventView::retryable(
  self : LunaFailedEventView,
) -> Bool raise LunaEventError {
  luna_authenticate_view(self.owner, self.epoch, LunaFailedEventSlot).retryable
}

///|
pub fn LunaFailedEventView::code_length(
  self : LunaFailedEventView,
) -> Int raise LunaEventError {
  luna_authenticate_view(self.owner, self.epoch, LunaFailedEventSlot).payload_length
}

///|
pub fn LunaFailedEventView::copy_code_to(
  self : LunaFailedEventView,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> Int raise LunaEventError {
  luna_copy_payload(
    luna_authenticate_view(self.owner, self.epoch, LunaFailedEventSlot),
    destination,
    destination_offset,
  )
}
