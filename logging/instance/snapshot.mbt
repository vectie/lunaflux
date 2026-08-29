///|
pub fn LunaInstanceLog::snapshot(
  self : LunaInstanceLog,
) -> LunaInstanceLogSnapshot raise LunaInstanceLogError {
  if self.snapshot_generation == 0xffffffffffffffffUL {
    raise LunaInstanceLogSnapshotGenerationExhausted
  }
  for index in 0..<self.length {
    let source = self.logical_index(index)
    self.snapshot_events[index] = self.events[source]
    self.snapshot_reasons[index] = self.reasons[source]
    self.snapshot_timestamps[index] = self.timestamps[source]
    self.snapshot_counts[index] = self.counts[source]
    self.snapshot_durations[index] = self.durations[source]
  }
  self.snapshot_length = self.length
  self.snapshot_dropped_count = self.dropped_count
  let generation = self.snapshot_generation + 1UL
  self.snapshot_generation = generation
  { owner: self, generation }
}

///|
pub fn LunaInstanceLogSnapshot::is_live(self : LunaInstanceLogSnapshot) -> Bool {
  self.generation != 0UL && self.owner.snapshot_generation == self.generation
}

///|
fn LunaInstanceLogSnapshot::require_live(
  self : LunaInstanceLogSnapshot,
) -> Unit raise LunaInstanceLogError {
  if !self.is_live() {
    raise LunaInstanceLogStaleSnapshot
  }
}

///|
fn LunaInstanceLogSnapshot::require_index(
  self : LunaInstanceLogSnapshot,
  index : Int,
) -> Unit raise LunaInstanceLogError {
  self.require_live()
  if index < 0 || index >= self.owner.snapshot_length {
    raise LunaInstanceLogInvalid(LunaInstanceLogIndexRule)
  }
}

///|
pub fn LunaInstanceLogSnapshot::length(
  self : LunaInstanceLogSnapshot,
) -> Int raise LunaInstanceLogError {
  self.require_live()
  self.owner.snapshot_length
}

///|
pub fn LunaInstanceLogSnapshot::dropped_count(
  self : LunaInstanceLogSnapshot,
) -> UInt64 raise LunaInstanceLogError {
  self.require_live()
  self.owner.snapshot_dropped_count
}

///|
pub fn LunaInstanceLogSnapshot::event_at(
  self : LunaInstanceLogSnapshot,
  index : Int,
) -> LunaInstanceLogEvent raise LunaInstanceLogError {
  self.require_index(index)
  luna_instance_log_event_from_code(self.owner.snapshot_events[index])
}

///|
pub fn LunaInstanceLogSnapshot::reason_at(
  self : LunaInstanceLogSnapshot,
  index : Int,
) -> LunaInstanceLogReason raise LunaInstanceLogError {
  self.require_index(index)
  luna_instance_log_reason_from_code(self.owner.snapshot_reasons[index])
}

///|
pub fn LunaInstanceLogSnapshot::timestamp_millis_at(
  self : LunaInstanceLogSnapshot,
  index : Int,
) -> UInt64 raise LunaInstanceLogError {
  self.require_index(index)
  self.owner.snapshot_timestamps[index]
}

///|
pub fn LunaInstanceLogSnapshot::count_at(
  self : LunaInstanceLogSnapshot,
  index : Int,
) -> Int raise LunaInstanceLogError {
  self.require_index(index)
  self.owner.snapshot_counts[index]
}

///|
pub fn LunaInstanceLogSnapshot::duration_millis_at(
  self : LunaInstanceLogSnapshot,
  index : Int,
) -> Int raise LunaInstanceLogError {
  self.require_index(index)
  self.owner.snapshot_durations[index]
}
