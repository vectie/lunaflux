///|
fn request_identity_tracker_capacity(maximum_rows : Int) -> Int {
  let required = maximum_rows * 2
  let mut capacity = 1
  while capacity < required {
    capacity = capacity * 2
  }
  capacity
}

///|
fn request_identity_start(value : UInt64, capacity : Int) -> Int {
  let folded = (value ^ (value >> 32)).to_uint()
  (folded & (capacity - 1).reinterpret_as_uint()).reinterpret_as_int()
}

///|
fn SchedulePlanBuffer::has_request_identity(
  self : SchedulePlanBuffer,
  request_id : @inference.RequestId,
) -> Bool {
  let value = request_id.value()
  let capacity = self.request_identity_keys.length()
  let start = request_identity_start(value, capacity)
  for probe in 0..<capacity {
    let index = (start + probe) & (capacity - 1)
    if self.request_identity_generations[index] !=
      self.identity_tracker_generation {
      return false
    }
    if self.request_identity_keys[index] == value {
      return true
    }
  }
  false
}

///|
fn SchedulePlanBuffer::has_completion_slot_identity(
  self : SchedulePlanBuffer,
  completion_slot : CompletionSlot,
) -> Bool {
  self.completion_slot_generations[completion_slot.value()] ==
  self.identity_tracker_generation
}

///|
fn SchedulePlanBuffer::track_request_identity(
  self : SchedulePlanBuffer,
  request_id : @inference.RequestId,
) -> Unit {
  let value = request_id.value()
  let capacity = self.request_identity_keys.length()
  let start = request_identity_start(value, capacity)
  for probe in 0..<capacity {
    let index = (start + probe) & (capacity - 1)
    if self.request_identity_generations[index] !=
      self.identity_tracker_generation {
      self.request_identity_keys[index] = value
      self.request_identity_generations[index] = self.identity_tracker_generation
      return
    }
    if self.request_identity_keys[index] == value {
      abort("plan request-identity tracker duplicate invariant")
    }
  }
  abort("plan request-identity tracker capacity invariant")
}

///|
fn SchedulePlanBuffer::track_row_identity(
  self : SchedulePlanBuffer,
  request_id : @inference.RequestId,
  completion_slot : CompletionSlot,
) -> Unit {
  let slot = completion_slot.value()
  if self.completion_slot_generations[slot] == self.identity_tracker_generation {
    abort("plan completion-slot tracker duplicate invariant")
  }
  self.track_request_identity(request_id)
  self.completion_slot_generations[slot] = self.identity_tracker_generation
}

///|
fn SchedulePlanBuffer::advance_identity_tracker_generation(
  self : SchedulePlanBuffer,
) -> Unit {
  if self.identity_tracker_generation == 0xffffffffffffffffUL {
    for index in 0..<self.request_identity_generations.length() {
      self.request_identity_generations[index] = 0UL
    }
    for index in 0..<self.completion_slot_generations.length() {
      self.completion_slot_generations[index] = 0UL
    }
    self.identity_tracker_generation = 1UL
  } else {
    self.identity_tracker_generation += 1UL
  }
}

///|
fn SchedulePlanBuffer::rebuild_identity_tracker(
  self : SchedulePlanBuffer,
) -> Unit {
  self.advance_identity_tracker_generation()
  for index in 0..<self.prefill_count {
    self.track_row_identity(
      self.prefill_request_ids[index],
      self.prefill_completion_slots[index],
    )
  }
  for index in 0..<self.decode_count {
    self.track_row_identity(
      self.decode_request_ids[index],
      self.decode_completion_slots[index],
    )
  }
}
