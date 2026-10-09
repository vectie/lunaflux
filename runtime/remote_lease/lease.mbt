///|
pub(all) enum RemoteLeaseStatus {
  Active
  Expired
  ClockRegressed
  ClockOverflow
  InvalidRenewal
  TransportLost
} derive(Debug, Eq)

///|
pub(all) suberror RemoteLeaseError {
  InvalidConfiguration
} derive(Debug, Eq)

///|
/// Pure value transition for one remote worker generation's local watchdog.
/// A caller replaces its state with each returned value and fences its owned
/// resources whenever status is not Active. This type performs no I/O/cleanup.
#valtype
pub struct RemoteLease {
  priv generation : UInt64
  priv duration_millis : UInt64
  priv last_tick : UInt64
  priv deadline : UInt64
  priv renewal_sequence : UInt64
  priv status : RemoteLeaseStatus
} derive(Debug, Eq)

///|
/// Start from this worker's monotonic millisecond clock, never a peer timestamp.
pub fn RemoteLease::start(
  generation~ : UInt64,
  duration_millis~ : UInt64,
  local_now_millis~ : UInt64,
) -> RemoteLease raise RemoteLeaseError {
  if generation == 0UL ||
    duration_millis == 0UL ||
    duration_millis > 300000UL ||
    local_now_millis > 0xffffffffffffffffUL - duration_millis {
    raise InvalidConfiguration
  }
  {
    generation,
    duration_millis,
    last_tick: local_now_millis,
    deadline: local_now_millis + duration_millis,
    renewal_sequence: 0UL,
    status: Active,
  }
}

///|
pub fn RemoteLease::status(self : RemoteLease) -> RemoteLeaseStatus {
  self.status
}

///|
pub fn RemoteLease::deadline_millis(self : RemoteLease) -> UInt64 {
  self.deadline
}

///|
pub fn RemoteLease::generation_value(self : RemoteLease) -> UInt64 {
  self.generation
}

///|
pub fn RemoteLease::renewal_sequence_value(self : RemoteLease) -> UInt64 {
  self.renewal_sequence
}

///|
/// Poll independently of NCCL progress. Expiration is inclusive at deadline.
pub fn RemoteLease::poll(
  self : RemoteLease,
  local_now_millis : UInt64,
) -> RemoteLease {
  if self.status != Active {
    return self
  }
  if local_now_millis < self.last_tick {
    return { ..self, status: ClockRegressed, }
  }
  {
    ..self,
    last_tick: local_now_millis,
    status: if local_now_millis >= self.deadline {
      Expired
    } else {
      Active
    },
  }
}

///|
/// An authenticated, strictly next heartbeat may extend only a still-live
/// lease. A delayed heartbeat cannot resurrect an expired generation.
/// The sequence is generation-local; heartbeat messages carry no clock value.
pub fn RemoteLease::renew(
  self : RemoteLease,
  generation~ : UInt64,
  sequence~ : UInt64,
  local_now_millis~ : UInt64,
) -> RemoteLease {
  let observed = self.poll(local_now_millis)
  if observed.status != Active {
    return observed
  }
  if generation != self.generation ||
    self.renewal_sequence == 0xffffffffffffffffUL ||
    sequence != self.renewal_sequence + 1UL {
    return { ..observed, status: InvalidRenewal, }
  }
  if local_now_millis > 0xffffffffffffffffUL - self.duration_millis {
    return { ..observed, status: ClockOverflow, }
  }
  {
    ..observed,
    deadline: local_now_millis + self.duration_millis,
    renewal_sequence: sequence,
  }
}

///|
pub fn RemoteLease::transport_lost(self : RemoteLease) -> RemoteLease {
  if self.status != Active {
    self
  } else {
    { ..self, status: TransportLost, }
  }
}

///|
pub extend RemoteLeaseStatus with Eq::{equal, not_equal}

///|
pub extend RemoteLeaseStatus with @debug.Debug::{to_repr}

///|
pub extend RemoteLeaseError with Eq::{equal, not_equal}

///|
pub extend RemoteLeaseError with @debug.Debug::{to_repr}

///|
pub extend RemoteLease with Eq::{equal, not_equal}

///|
pub extend RemoteLease with @debug.Debug::{to_repr}
