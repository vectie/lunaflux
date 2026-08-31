///|
fn luna_online_tcp_error(rule : LunaOnlineTcpRule) -> LunaOnlineTcpError {
  LunaOnlineTcpFailed(rule)
}

///|
fn LunaOnlineTcpLimits::is_valid(self : LunaOnlineTcpLimits) -> Bool {
  self.read_chunk_bytes > 0 &&
  self.read_chunk_bytes <= LUNA_ONLINE_TCP_MAX_SCRATCH_BYTES &&
  self.write_chunk_bytes > 0 &&
  self.write_chunk_bytes <= LUNA_ONLINE_TCP_MAX_SCRATCH_BYTES &&
  self.accept_timeout_millis > 0 &&
  self.accept_timeout_millis <= LUNA_ONLINE_TCP_MAX_TIMEOUT_MILLIS &&
  self.input_idle_timeout_millis > 0 &&
  self.input_idle_timeout_millis <= LUNA_ONLINE_TCP_MAX_TIMEOUT_MILLIS &&
  self.write_timeout_millis > 0 &&
  self.write_timeout_millis <= LUNA_ONLINE_TCP_MAX_TIMEOUT_MILLIS &&
  self.reactor_transition_budget > 0 &&
  self.reactor_transition_budget <= LUNA_ONLINE_TCP_MAX_TRANSITIONS
}

///|
fn LunaOnlineTcpLimits::fits_transport_wait(
  self : LunaOnlineTcpLimits,
  maximum_wait_millis : Int,
) -> Bool {
  maximum_wait_millis > 0 &&
  self.input_idle_timeout_millis <= maximum_wait_millis &&
  self.write_timeout_millis <= maximum_wait_millis
}

///| Prove that request-owned transport waits dominate both I/O timeouts.

///|
/// This is the same invariant enforced by endpoint/server activation and is
/// exposed for pure instance admission so policy validation cannot drift.
pub fn LunaOnlineTcpLimits::require_fits_transport_wait(
  self : LunaOnlineTcpLimits,
  maximum_wait_millis : Int,
) -> Unit raise LunaOnlineTcpError {
  if !self.is_valid() || !self.fits_transport_wait(maximum_wait_millis) {
    raise luna_online_tcp_error(LunaOnlineTcpLimits)
  }
}

///|
fn luna_online_tcp_safe_wait(
  configured_millis : Int,
  remaining_millis : Int,
) -> Int {
  if configured_millis <= 0 || remaining_millis <= 1 {
    return 0
  }
  Int::min(configured_millis, remaining_millis - 1)
}

///|
pub fn LunaOnlineTcpLimits::new(
  read_chunk_bytes~ : Int,
  write_chunk_bytes~ : Int,
  accept_timeout_millis~ : Int,
  input_idle_timeout_millis~ : Int,
  write_timeout_millis~ : Int,
  reactor_transition_budget~ : Int,
) -> LunaOnlineTcpLimits raise LunaOnlineTcpError {
  let limits = {
    read_chunk_bytes,
    write_chunk_bytes,
    accept_timeout_millis,
    input_idle_timeout_millis,
    write_timeout_millis,
    reactor_transition_budget,
  }
  if !limits.is_valid() {
    raise luna_online_tcp_error(LunaOnlineTcpLimits)
  }
  limits
}

///|
pub fn LunaOnlineTcpLimits::read_chunk_bytes(self : LunaOnlineTcpLimits) -> Int {
  self.read_chunk_bytes
}

///|
pub fn LunaOnlineTcpLimits::write_chunk_bytes(
  self : LunaOnlineTcpLimits,
) -> Int {
  self.write_chunk_bytes
}

///|
pub fn LunaOnlineTcpLimits::reactor_transition_budget(
  self : LunaOnlineTcpLimits,
) -> Int {
  self.reactor_transition_budget
}

///|
const LUNA_ONLINE_POOL_POLL_MAX_MILLIS : Int = 1

///|
/// Returns the exact authenticated transport ceiling used by pool-owned short
/// socket polls. A pool shares one cooperative reactor across listener,
/// connections, and service progress, so an idle socket must never retain that
/// reactor for an operational timeout. The one-millisecond quantum remains
/// within every authenticated accept, input-idle, and write authority.
pub fn LunaOnlineTcpLimits::pool_poll_timeout_millis(
  self : LunaOnlineTcpLimits,
) -> Int {
  Int::min(
    LUNA_ONLINE_POOL_POLL_MAX_MILLIS,
    Int::min(
      self.accept_timeout_millis,
      Int::min(self.input_idle_timeout_millis, self.write_timeout_millis),
    ),
  )
}
