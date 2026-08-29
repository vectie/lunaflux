///|
fn luna_instance_log_requirements(
  capacity : Int,
) -> (UInt64, UInt64) raise LunaInstanceLogError {
  if capacity <= 0 || capacity > LUNA_INSTANCE_LOG_MAX_CAPACITY {
    raise LunaInstanceLogInvalid(LunaInstanceLogCapacityRule)
  }
  let cells = capacity.to_uint64()
  (cells * 8UL, cells * 2UL)
}

///|
/// Exact current-ring plus snapshot integer cells.
pub fn LunaInstanceLog::required_int_cells(
  capacity : Int,
) -> UInt64 raise LunaInstanceLogError {
  luna_instance_log_requirements(capacity).0
}

///|
/// Exact current-ring plus snapshot timestamp cells.
pub fn LunaInstanceLog::required_uint64_cells(
  capacity : Int,
) -> UInt64 raise LunaInstanceLogError {
  luna_instance_log_requirements(capacity).1
}

///|
pub fn LunaInstanceLog::new(
  capacity : Int,
) -> LunaInstanceLog raise LunaInstanceLogError {
  ignore(luna_instance_log_requirements(capacity))
  {
    events: FixedArray::make(capacity, 0),
    reasons: FixedArray::make(capacity, 0),
    timestamps: FixedArray::make(capacity, 0UL),
    counts: FixedArray::make(capacity, 0),
    durations: FixedArray::make(capacity, 0),
    snapshot_events: FixedArray::make(capacity, 0),
    snapshot_reasons: FixedArray::make(capacity, 0),
    snapshot_timestamps: FixedArray::make(capacity, 0UL),
    snapshot_counts: FixedArray::make(capacity, 0),
    snapshot_durations: FixedArray::make(capacity, 0),
    head: 0,
    length: 0,
    dropped_count: 0UL,
    has_timestamp: false,
    last_timestamp: 0UL,
    snapshot_length: 0,
    snapshot_dropped_count: 0UL,
    snapshot_generation: 0UL,
  }
}

///|
pub fn LunaInstanceLog::capacity(self : LunaInstanceLog) -> Int {
  self.events.length()
}

///|
pub fn LunaInstanceLog::length(self : LunaInstanceLog) -> Int {
  self.length
}

///|
pub fn LunaInstanceLog::dropped_count(self : LunaInstanceLog) -> UInt64 {
  self.dropped_count
}

///|
fn LunaInstanceLog::logical_index(
  self : LunaInstanceLog,
  index : Int,
) -> Int raise LunaInstanceLogError {
  if index < 0 || index >= self.length {
    raise LunaInstanceLogInvalid(LunaInstanceLogIndexRule)
  }
  let raw = self.head + index
  if raw >= self.events.length() {
    raw - self.events.length()
  } else {
    raw
  }
}

///|
/// Records one scalar-only event, overwriting the oldest entry when full.
pub fn LunaInstanceLog::record(
  self : LunaInstanceLog,
  event : LunaInstanceLogEvent,
  reason : LunaInstanceLogReason,
  timestamp_millis~ : UInt64,
  count~ : Int,
  duration_millis~ : Int,
) -> Unit raise LunaInstanceLogError {
  if self.has_timestamp && timestamp_millis < self.last_timestamp {
    raise LunaInstanceLogInvalid(LunaInstanceLogTimestampRule)
  }
  if count < 0 || count > LUNA_INSTANCE_LOG_MAX_COUNT {
    raise LunaInstanceLogInvalid(LunaInstanceLogCountRule)
  }
  if duration_millis < 0 ||
    duration_millis > LUNA_INSTANCE_LOG_MAX_DURATION_MILLIS {
    raise LunaInstanceLogInvalid(LunaInstanceLogDurationRule)
  }
  let index = if self.length < self.events.length() {
    let value = self.head + self.length
    self.length += 1
    if value >= self.events.length() {
      value - self.events.length()
    } else {
      value
    }
  } else {
    let value = self.head
    self.head += 1
    if self.head == self.events.length() {
      self.head = 0
    }
    if self.dropped_count != 0xffffffffffffffffUL {
      self.dropped_count += 1UL
    }
    value
  }
  self.events[index] = luna_instance_log_event_code(event)
  self.reasons[index] = luna_instance_log_reason_code(reason)
  self.timestamps[index] = timestamp_millis
  self.counts[index] = count
  self.durations[index] = duration_millis
  self.has_timestamp = true
  self.last_timestamp = timestamp_millis
}

///|
pub fn LunaInstanceLog::event_at(
  self : LunaInstanceLog,
  index : Int,
) -> LunaInstanceLogEvent raise LunaInstanceLogError {
  luna_instance_log_event_from_code(self.events[self.logical_index(index)])
}

///|
pub fn LunaInstanceLog::reason_at(
  self : LunaInstanceLog,
  index : Int,
) -> LunaInstanceLogReason raise LunaInstanceLogError {
  luna_instance_log_reason_from_code(self.reasons[self.logical_index(index)])
}

///|
pub fn LunaInstanceLog::timestamp_millis_at(
  self : LunaInstanceLog,
  index : Int,
) -> UInt64 raise LunaInstanceLogError {
  self.timestamps[self.logical_index(index)]
}

///|
pub fn LunaInstanceLog::count_at(
  self : LunaInstanceLog,
  index : Int,
) -> Int raise LunaInstanceLogError {
  self.counts[self.logical_index(index)]
}

///|
pub fn LunaInstanceLog::duration_millis_at(
  self : LunaInstanceLog,
  index : Int,
) -> Int raise LunaInstanceLogError {
  self.durations[self.logical_index(index)]
}

///|
/// Drops the current logical ring without touching the snapshot slot.
pub fn LunaInstanceLog::reset(self : LunaInstanceLog) -> Unit {
  self.head = 0
  self.length = 0
  self.dropped_count = 0UL
  self.has_timestamp = false
  self.last_timestamp = 0UL
}
