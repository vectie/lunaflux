///|
fn LunaRequestPreparationPool::abort_lane_resources(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit {
  ignore(self)
  if !lane.framed_view.is_empty() {
    try! lane.framed_view[0].release()
    lane.framed_view.clear()
  }
  if !lane.typed_handoff.is_empty() {
    try! lane.typed_handoff[0].release()
    lane.typed_handoff.clear()
  }
  if !lane.framed_work.is_empty() {
    try! lane.framed_work[0].abort()
    lane.framed_work.clear()
  }
  match lane.tokenizer_input_write {
    Some(write) => {
      try! write.abort()
      lane.tokenizer_input_write = None
    }
    None => ()
  }
  match lane.tokenizer_work {
    Some(work) => {
      try! work.abort()
      lane.tokenizer_work = None
    }
    None => ()
  }
  match lane.token_write {
    Some(write) => {
      try! write.abort()
      lane.token_write = None
    }
    None => ()
  }
  match lane.token_lease {
    Some(lease) => {
      try! lease.release()
      lane.token_lease = None
    }
    None => ()
  }
  match lane.output_work {
    Some(work) => {
      try! work.abort()
      lane.output_work = None
    }
    None => ()
  }
  match lane.output_lease {
    Some(lease) => {
      try! lease.release()
      lane.output_lease = None
    }
    None => ()
  }
  match lane.claim {
    Some(claim) => {
      try! claim.release_resources_only()
      return
    }
    None => ()
  }
  if !lane.semantic_write.is_empty() {
    try! lane.semantic_write[0].abort()
    lane.semantic_write.clear()
  }
  if !lane.semantic_work.is_empty() {
    try! lane.semantic_work[0].abort()
    lane.semantic_work.clear()
  }
  if !lane.semantic_lease.is_empty() {
    try! lane.semantic_lease[0].release()
    lane.semantic_lease.clear()
  }
}

///|
fn LunaRequestPreparationPool::fail_lane(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
  failure : RequestAdmissionError,
) -> Unit {
  self.abort_lane_resources(lane)
  lane.claim = None
  lane.failure = Some(failure)
  lane.phase = LUNA_PREPARATION_FAILED
}
