///|
fn LunaRequestPreparationWork::require_lane(
  self : LunaRequestPreparationWork,
) -> LunaRequestPreparationLane raise RequestAdmissionError {
  if self.lane < 0 || self.lane >= self.pool.lanes.length() {
    raise Invalid(LunaPreparedLifecycle)
  }
  let lane = self.pool.lanes[self.lane]
  if self.generation == 0UL ||
    lane.generation != self.generation ||
    lane.phase == LUNA_PREPARATION_IDLE ||
    lane.phase == LUNA_PREPARATION_PREPARED ||
    lane.phase == LUNA_PREPARATION_CLAIMED {
    raise Invalid(LunaPreparedLifecycle)
  }
  lane
}

///|
pub fn LunaRequestPreparationWork::state(
  self : LunaRequestPreparationWork,
) -> LunaRequestPreparationWorkState raise RequestAdmissionError {
  match self.require_lane().phase {
    LUNA_PREPARATION_READY => LunaRequestPreparationReady
    LUNA_PREPARATION_FAILED => LunaRequestPreparationFailed
    LUNA_PREPARATION_CANCEL_REQUESTED => LunaRequestPreparationCancelRequested
    _ => LunaRequestPreparationPending
  }
}

///|
pub fn LunaRequestPreparationWork::last_work_units(
  self : LunaRequestPreparationWork,
) -> Int raise RequestAdmissionError {
  self.require_lane().last_work_units
}

///|
pub fn LunaRequestPreparationWork::total_work_units(
  self : LunaRequestPreparationWork,
) -> UInt64 raise RequestAdmissionError {
  self.require_lane().total_work_units
}

///|
pub fn LunaRequestPreparationWork::failure(
  self : LunaRequestPreparationWork,
) -> RequestAdmissionError raise RequestAdmissionError {
  let lane = self.require_lane()
  if lane.phase != LUNA_PREPARATION_FAILED {
    raise Invalid(LunaPreparedLifecycle)
  }
  match lane.failure {
    Some(value) => value
    None => abort("Luna preparation failure invariant")
  }
}

///|
pub fn LunaRequestPreparationWork::cancel(
  self : LunaRequestPreparationWork,
) -> Unit raise RequestAdmissionError {
  let lane = self.require_lane()
  match lane.phase {
    LUNA_PREPARATION_READY | LUNA_PREPARATION_FAILED => {
      self.pool.abort_lane_resources(lane)
      self.pool.release_lane(self.lane, self.generation)
    }
    LUNA_PREPARATION_CANCEL_REQUESTED => ()
    _ => lane.phase = LUNA_PREPARATION_CANCEL_REQUESTED
  }
}

///|
pub fn LunaRequestPreparationWork::discard(
  self : LunaRequestPreparationWork,
) -> Unit raise RequestAdmissionError {
  self.cancel()
}

///|
pub fn LunaRequestPreparationWork::take_prepared(
  self : LunaRequestPreparationWork,
) -> LunaPreparedRequest raise RequestAdmissionError {
  let lane = self.require_lane()
  if lane.phase != LUNA_PREPARATION_READY {
    raise Invalid(LunaPreparedLifecycle)
  }
  validate_pool_time(
    lane.receipt_at_millis,
    lane.require_current_deadline(),
    (self.pool.clock_read)(),
  ) catch {
    error => {
      self.pool.fail_lane(lane, error)
      raise error
    }
  }
  let claim = match lane.claim {
    Some(value) => value
    None => abort("Luna ready-claim invariant")
  }
  lane.phase = LUNA_PREPARATION_PREPARED
  { claim: Some(claim) }
}

///|
pub fn LunaRequestPreparationAdmission::kind(
  self : LunaRequestPreparationAdmission,
) -> LunaRequestPreparationAdmissionKind {
  match self.disposition {
    0 => LunaRequestPreparationAdmitted
    1 => LunaRequestPreparationSaturated
    2 => LunaRequestPreparationDraining
    _ => abort("Luna preparation admission invariant")
  }
}

///|
pub fn LunaRequestPreparationAdmission::consumed_bytes(
  self : LunaRequestPreparationAdmission,
) -> Int {
  self.consumed
}

///|
pub fn LunaRequestPreparationAdmission::work(
  self : LunaRequestPreparationAdmission,
) -> LunaRequestPreparationWork raise RequestAdmissionError {
  if self.disposition != 0 || self.lane < 0 || self.generation == 0UL {
    raise Invalid(LunaPreparedLifecycle)
  }
  { pool: self.pool, lane: self.lane, generation: self.generation }
}
