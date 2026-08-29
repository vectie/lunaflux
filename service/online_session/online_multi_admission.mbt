///|
fn luna_online_multi_admission(
  kind : LunaOnlineInstanceAdmissionKind,
  lane : Int,
  generation : UInt64,
) -> LunaOnlineInstanceAdmission {
  {
    disposition: match kind {
      LunaOnlineInstanceAdmitted => 0
      LunaOnlineInstanceBusy => 1
      LunaOnlineInstanceDraining => 2
    },
    lane,
    request_epoch: generation,
  }
}

///|
fn LunaOnlineInstance::require_multi_ticket(
  self : LunaOnlineInstance,
  ticket : LunaOnlineRequestTicket,
) -> LunaOnlineMultiLane raise LunaOnlineInstanceError {
  if ticket.lane < 0 ||
    ticket.lane >= self.multi.lanes.length() ||
    ticket.epoch == 0UL {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineRequestEpoch))
  }
  let lane = self.multi.lanes[ticket.lane]
  if lane.generation != ticket.epoch ||
    lane.phase == LunaInstanceIdlePhase ||
    lane.claims.length() != 1 ||
    lane.worker_requests.length() != 1 {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineRequestEpoch))
  }
  lane
}

///|
fn LunaOnlineInstance::require_active_multi_ticket(
  self : LunaOnlineInstance,
  ticket : LunaOnlineRequestTicket,
) -> LunaOnlineMultiLane raise LunaOnlineInstanceError {
  let lane = self.require_multi_ticket(ticket)
  if lane.phase != LunaRequestActivePhase {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineLifecycle))
  }
  lane
}

///|
#warnings("-unused_value")
fn LunaOnlineInstance::multi_claim_owner(
  self : LunaOnlineInstance,
  lane : Int,
) -> @request_admission.LunaPreparedRequestClaim {
  if lane < 0 ||
    lane >= self.multi.lanes.length() ||
    self.multi.lanes[lane].claims.length() != 1 {
    abort("Luna online multi claim invariant")
  }
  self.multi.lanes[lane].claims[0]
}

///|
#warnings("-unused_value")
fn LunaOnlineInstance::multi_worker_request(
  self : LunaOnlineInstance,
  lane : Int,
) -> @worker_service.LunaOnlineWorkerRequest {
  if lane < 0 ||
    lane >= self.multi.lanes.length() ||
    self.multi.lanes[lane].worker_requests.length() != 1 {
    abort("Luna online multi worker request invariant")
  }
  self.multi.lanes[lane].worker_requests[0]
}

///|
fn LunaOnlineInstance::mark_multi_event_published(
  self : LunaOnlineInstance,
  lane : Int,
  kind : SessionEventKind,
) -> Unit {
  if self.multi.event_lane >= 0 || !self.events.has_event() {
    abort("Luna online multi event credit invariant")
  }
  self.multi.event_lane = lane
  self.mark_event_published(kind)
}

///|
#warnings("-unused_value")
fn LunaOnlineInstance::multi_event_ticket(
  self : LunaOnlineInstance,
) -> LunaOnlineRequestTicket {
  let lane = self.multi.event_lane
  if lane < 0 || lane >= self.multi.lanes.length() {
    abort("Luna online multi event lane invariant")
  }
  { lane, epoch: self.multi.lanes[lane].generation }
}

///|
#warnings("-unused_value")
fn LunaOnlineInstance::multi_active_request_count(
  self : LunaOnlineInstance,
) -> Int {
  self.multi.active_count
}

///|
fn LunaOnlineInstance::peek_free_multi_lane(self : LunaOnlineInstance) -> Int {
  if self.multi.free_count == 0 {
    return -1
  }
  let lane = self.multi.free_lanes[self.multi.free_head]
  if lane < 0 ||
    lane >= self.multi.lanes.length() ||
    self.multi.lanes[lane].phase != LunaInstanceIdlePhase {
    abort("Luna online multi free ring invariant")
  }
  lane
}

///|
fn LunaOnlineInstance::commit_free_multi_lane(
  self : LunaOnlineInstance,
  lane_index : Int,
) -> Unit {
  if self.multi.free_count <= 0 ||
    self.multi.free_lanes[self.multi.free_head] != lane_index {
    abort("Luna online multi free ring commit invariant")
  }
  self.multi.free_head = (self.multi.free_head + 1) %
    self.multi.free_lanes.length()
  self.multi.free_count -= 1
}

///|
#warnings("-unused_value")
fn LunaOnlineInstance::begin_multi(
  self : LunaOnlineInstance,
  prepared : @request_admission.LunaPreparedRequest,
) -> LunaOnlineInstanceAdmission raise LunaOnlineInstanceError {
  if self.drain_requested ||
    self.recovery_close ||
    self.phase == LunaInstanceRecoveringTerminalPhase ||
    self.phase == LunaInstanceClosingTerminalPhase ||
    self.phase == LunaInstanceRetryingTerminalClosePhase ||
    self.phase == LunaInstanceDrainingPhase ||
    self.phase == LunaInstanceRetryingEmptyClosePhase ||
    self.phase == LunaInstanceClosedPhase {
    return luna_online_multi_admission(LunaOnlineInstanceDraining, -1, 0UL)
  }
  if self.claim is Some(_) ||
    (self.phase == LunaRequestActivePhase && self.multi.active_count == 0) ||
    self.events.has_event() ||
    self.multi.active_count >= self.multi.lanes.length() {
    return luna_online_multi_admission(LunaOnlineInstanceBusy, -1, 0UL)
  }
  if self.request_epoch == 0xffffffffffffffffUL {
    self.drain_requested = true
    return luna_online_multi_admission(LunaOnlineInstanceDraining, -1, 0UL)
  }
  let stream_preference = prepared.stream_preference() catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure)
  }
  if stream_preference != Streaming {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineStreaming))
  }
  let model_identity = prepared.model_identity() catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure)
  }
  if model_identity != self.expected_model {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineModelIdentity))
  }
  let tokenizer_digest = prepared.tokenizer_digest() catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure)
  }
  if tokenizer_digest != self.expected_tokenizer {
    raise LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure)
  }
  let inference_limits = prepared.inference_limits() catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure)
  }
  if inference_limits != self.inference {
    raise LunaOnlineInstanceFailed(
      LunaOnlineInvalid(LunaOnlineInferenceEnvelope),
    )
  }
  let effective = prepared.effective_limits() catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure)
  }
  if !self.events.has_epoch_capacity(effective.max_new_tokens() + 3) {
    self.drain_requested = true
    return luna_online_multi_admission(LunaOnlineInstanceDraining, -1, 0UL)
  }
  let lease = self.lease_owner()
  let lower_active = try! lease.active_request_count()
  if lower_active >= lease.request_capacity() {
    return luna_online_multi_admission(LunaOnlineInstanceBusy, -1, 0UL)
  }
  let lane_index = self.peek_free_multi_lane()
  if lane_index < 0 {
    abort("Luna online multi free lane invariant")
  }
  let lane = self.multi.lanes[lane_index]
  let input_tokens = prepared.input_token_count() catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure)
  }
  self.events.publish_accepted(model_identity, effective) catch {
    LunaEventInvalid(LunaEventEpoch) => {
      self.drain_requested = true
      return luna_online_multi_admission(LunaOnlineInstanceDraining, -1, 0UL)
    }
    _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
  }
  let claim = prepared.take_claim() catch {
    _ => {
      try! self.events.discard()
      raise LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure)
    }
  }
  let lower = lease.try_admit_request(claim.scheduler_request()) catch {
    _ => {
      try! claim.release()
      try! self.events.discard()
      raise LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure)
    }
  }
  if lower.kind() != LunaOnlineWorkerAdmitted {
    try! claim.release()
    try! self.events.discard()
    raise LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure)
  }
  let worker_request = lower.request() catch {
    _ => abort("Luna online multi lower admission invariant")
  }
  let generation = self.request_epoch + 1UL
  lane.claims.push(claim)
  lane.worker_requests.push(worker_request)
  lane.generation = generation
  lane.input_tokens = input_tokens
  lane.cached_input_tokens = 0
  lane.accepted_acked = false
  lane.output_tokens = 0
  lane.final_tail_length = 0
  lane.finish_kind = FinishPending
  lane.string_cut = NoStringCut
  lane.terminal_cut = NoTerminalCut
  lane.deferred_cancel = false
  lane.failure_kind = NoLatchedFailure
  lane.restarted_failure_terminal_pending = false
  lane.recovery_close = false
  lane.phase = LunaRequestActivePhase
  self.multi.routes.bind(try! worker_request.publication_route(), lane_index)
  self.commit_free_multi_lane(lane_index)
  self.request_epoch = generation
  self.multi.active_count += 1
  self.mark_multi_event_published(lane_index, AcceptedEvent)
  self.phase = LunaRequestActivePhase
  luna_online_multi_admission(
    LunaOnlineInstanceAdmitted,
    lane_index,
    generation,
  )
}
