///|
pub fn LunaOnlineOpenAIConnectionPoolLimits::new(
  connection_capacity~ : Int,
  output_buffer_bytes~ : Int,
  io_poll_timeout_millis~ : Int,
) -> LunaOnlineOpenAIConnectionPoolLimits raise LunaOnlineTcpError {
  if connection_capacity <= 1 ||
    connection_capacity > LUNA_ONLINE_TCP_MAX_TRANSITIONS ||
    output_buffer_bytes <= 0 ||
    output_buffer_bytes > LUNA_ONLINE_TCP_MAX_SCRATCH_BYTES ||
    io_poll_timeout_millis <= 0 ||
    io_poll_timeout_millis > LUNA_ONLINE_TCP_MAX_TIMEOUT_MILLIS {
    raise luna_online_tcp_error(LunaOnlineTcpLimits)
  }
  { connection_capacity, output_buffer_bytes, io_poll_timeout_millis }
}

///|
pub fn LunaOnlineOpenAIConnectionPoolLimits::connection_capacity(
  self : LunaOnlineOpenAIConnectionPoolLimits,
) -> Int {
  self.connection_capacity
}

///|
pub fn LunaOnlineOpenAIConnectionPoolLimits::output_buffer_bytes(
  self : LunaOnlineOpenAIConnectionPoolLimits,
) -> Int {
  self.output_buffer_bytes
}

///|
fn LunaOnlineOpenAIConnectionPoolLimits::fits_write_chunk(
  self : LunaOnlineOpenAIConnectionPoolLimits,
  tcp_limits : LunaOnlineTcpLimits,
) -> Bool {
  tcp_limits.write_chunk_bytes >= self.output_buffer_bytes
}
