///|
fn LunaRequestPreparationLane::require_framed_work(
  self : LunaRequestPreparationLane,
) -> @framed_wire.LunaFramedRequestWork {
  if self.framed_work.length() != 1 {
    abort("Luna framed work invariant")
  }
  self.framed_work[0]
}

///|
fn LunaRequestPreparationLane::require_framed_view(
  self : LunaRequestPreparationLane,
) -> @framed_wire.LunaFramedRequestView {
  if self.framed_view.length() != 1 {
    abort("Luna framed view invariant")
  }
  self.framed_view[0]
}

///|
fn LunaRequestPreparationLane::require_current_deadline(
  self : LunaRequestPreparationLane,
) -> @inference.AdmissionDeadline {
  match self.deadline {
    Some(value) => value
    None =>
      match self.hard_deadline {
        Some(value) => value
        None => abort("Luna preparation hard-deadline invariant")
      }
  }
}

///|
fn luna_framed_admission_error(
  _error : @framed_wire.FramedWireError,
) -> RequestAdmissionError {
  Invalid(FrameContract)
}

///|
fn LunaRequestPreparationPool::offer_framed_lane(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
  source : FixedArray[Byte],
  source_offset : Int,
  length : Int,
  observed_at_millis : UInt64,
) -> Int raise RequestAdmissionError {
  lane.last_work_units = 0
  if length < 0 || source_offset < 0 || source_offset > source.length() - length {
    raise Invalid(FrameContract)
  }
  if length == 0 {
    return 0
  }
  validate_pool_time(
    lane.receipt_at_millis,
    lane.require_current_deadline(),
    observed_at_millis,
  ) catch {
    error => {
      self.fail_lane(lane, error)
      raise error
    }
  }
  let work = lane.require_framed_work()
  let mut consumed = 0
  let maximum = self.step_budget.work_units()
  while consumed < length && consumed < maximum {
    if lane.total_work_units >= self.work_limit.work_units() {
      if consumed > 0 {
        break
      }
      self.fail_lane(lane, Invalid(LunaPreparationWork))
      raise Invalid(LunaPreparationWork)
    }
    let state = work.state() catch {
      error => {
        let mapped = luna_framed_admission_error(error)
        self.fail_lane(lane, mapped)
        raise mapped
      }
    }
    match state {
      LunaFramedRequestReceivingPrefix | LunaFramedRequestReceivingBody => ()
      LunaFramedRequestFailed => {
        let mapped = Invalid(FrameContract)
        self.fail_lane(lane, mapped)
        raise mapped
      }
      _ => break
    }
    let accepted = work.offer(
      source,
      source_offset=source_offset + consumed,
      length=length - consumed,
    ) catch {
      error => {
        let mapped = luna_framed_admission_error(error)
        self.fail_lane(lane, mapped)
        raise mapped
      }
    }
    if accepted <= 0 {
      break
    }
    if state == LunaFramedRequestReceivingBody {
      let after = work.state() catch {
        error => {
          let mapped = luna_framed_admission_error(error)
          self.fail_lane(lane, mapped)
          raise mapped
        }
      }
      match after {
        LunaFramedRequestValidating | LunaFramedRequestReady =>
          lane.framed_receipt_complete = true
        LunaFramedRequestFailed => {
          let mapped = Invalid(FrameContract)
          self.fail_lane(lane, mapped)
          raise mapped
        }
        LunaFramedRequestReceivingPrefix | LunaFramedRequestReceivingBody => ()
      }
    }
    consumed += accepted
    lane.total_work_units += accepted.to_uint64()
  }
  lane.last_work_units = consumed
  consumed
}

///|
pub fn LunaRequestPreparationPool::try_begin_luna_framed(
  self : LunaRequestPreparationPool,
  source : FixedArray[Byte],
  source_offset~ : Int,
  length~ : Int,
) -> LunaRequestPreparationAdmission raise RequestAdmissionError {
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
  if length <= 0 ||
    source_offset < 0 ||
    source_offset > source.length() - length {
    raise Invalid(FrameContract)
  }
  let receipt = (self.clock_read)()
  let hard_budget = @inference.DeadlineBudget::from_milliseconds(
    self.inference.max_deadline_millis(),
    self.inference,
  ) catch {
    _ => raise Invalid(Deadline)
  }
  let hard_deadline = @inference.AdmissionDeadline::from_receipt(
    receipt, hard_budget,
  ) catch {
    _ => raise Invalid(Deadline)
  }
  let scanner = self.lanes[next_lane].framed_workspace.begin() catch {
    _ => raise Invalid(LunaPreparationPool)
  }
  let lane_index = self.pop_free_lane()
  let lane = self.lanes[lane_index]
  lane.generation += 1UL
  lane.receipt_at_millis = receipt
  lane.hard_deadline = Some(hard_deadline)
  lane.deadline = None
  lane.request = None
  lane.effective = None
  lane.framed_direct = true
  lane.framed_receipt_complete = false
  lane.framed_work.push(scanner)
  lane.phase = LUNA_PREPARATION_FRAMED_RECEIVING
  lane.failure = None
  lane.claim = None
  lane.last_work_units = 0
  lane.total_work_units = 0UL
  let consumed = self.offer_framed_lane(
    lane, source, source_offset, length, receipt,
  ) catch {
    error => {
      if lane.phase != LUNA_PREPARATION_FAILED {
        self.abort_lane_resources(lane)
      }
      self.reset_lane_to_idle(lane_index)
      raise error
    }
  }
  self.enqueue_active(lane_index)
  preparation_admission(
    self,
    LunaRequestPreparationAdmitted,
    lane_index,
    lane.generation,
    consumed,
  )
}

///|
pub fn LunaRequestPreparationWork::offer_luna_framed(
  self : LunaRequestPreparationWork,
  source : FixedArray[Byte],
  source_offset~ : Int,
  length~ : Int,
) -> Int raise RequestAdmissionError {
  let lane = self.require_lane()
  if !lane.framed_direct {
    raise Invalid(LunaPreparedLifecycle)
  }
  if lane.phase == LUNA_PREPARATION_FRAMED_SCANNING &&
    !lane.framed_receipt_complete {
    // Prefix/layout validation temporarily owns the scanner. The exact Work
    // still owns the incomplete receipt position, so expose transport
    // backpressure without inspecting caller storage or sampling the clock.
    lane.last_work_units = 0
    return 0
  }
  if lane.phase != LUNA_PREPARATION_FRAMED_RECEIVING {
    raise Invalid(LunaPreparedLifecycle)
  }
  lane.last_work_units = 0
  if length < 0 || source_offset < 0 || source_offset > source.length() - length {
    raise Invalid(FrameContract)
  }
  if length == 0 {
    return 0
  }
  let observed = (self.pool.clock_read)()
  self.pool.offer_framed_lane(lane, source, source_offset, length, observed)
}

///|
/// Reports whether the exact direct frame body has been fully consumed. This
/// does not imply that canonical validation, semantic import, or preparation
/// has completed.
pub fn LunaRequestPreparationWork::luna_framed_receipt_complete(
  self : LunaRequestPreparationWork,
) -> Bool raise RequestAdmissionError {
  let lane = self.require_lane()
  if lane.phase == LUNA_PREPARATION_FAILED ||
    lane.phase == LUNA_PREPARATION_CANCEL_REQUESTED {
    raise Invalid(LunaPreparedLifecycle)
  }
  if !lane.framed_direct {
    return false
  }
  lane.framed_receipt_complete
}

///|
/// Returns the exact remaining hard-receipt interval for an authenticated
/// incomplete direct frame. Expiry returns zero; stale, completed, and
/// object-form work reject without exposing the absolute deadline.
pub fn LunaRequestPreparationWork::luna_framed_receipt_remaining_millis(
  self : LunaRequestPreparationWork,
) -> Int raise RequestAdmissionError {
  let lane = self.require_lane()
  if !lane.framed_direct ||
    lane.framed_receipt_complete ||
    (
      lane.phase != LUNA_PREPARATION_FRAMED_RECEIVING &&
      lane.phase != LUNA_PREPARATION_FRAMED_SCANNING
    ) {
    raise Invalid(LunaPreparedLifecycle)
  }
  let deadline = match lane.hard_deadline {
    Some(value) => value.at_millis()
    None => abort("Luna preparation hard-deadline invariant")
  }
  let observed = (self.pool.clock_read)()
  if observed < lane.receipt_at_millis {
    raise Invalid(ReceiptTime)
  }
  if observed >= deadline {
    return 0
  }
  let remaining = deadline - observed
  if remaining > 0x7fffffffUL {
    raise Invalid(ReceiptTime)
  }
  remaining.to_int()
}
