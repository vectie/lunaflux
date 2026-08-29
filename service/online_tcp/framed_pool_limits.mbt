///|
pub fn LunaOnlineFramedConnectionPoolLimits::new(
  connection_capacity~ : Int,
  io_poll_timeout_millis~ : Int,
  request_work_units~ : Int,
  event_work_units~ : Int,
) -> LunaOnlineFramedConnectionPoolLimits raise LunaOnlineTcpError {
  if connection_capacity <= 0 ||
    connection_capacity > 65536 ||
    io_poll_timeout_millis <= 0 ||
    io_poll_timeout_millis > LUNA_ONLINE_TCP_MAX_TIMEOUT_MILLIS ||
    request_work_units <= 0 ||
    request_work_units > 65536 ||
    event_work_units <= 0 ||
    event_work_units > 65536 {
    raise luna_online_tcp_error(LunaOnlineTcpLimits)
  }
  {
    connection_capacity,
    io_poll_timeout_millis,
    request_work_units,
    event_work_units,
  }
}

///|
pub fn LunaOnlineFramedConnectionPoolLimits::connection_capacity(
  self : LunaOnlineFramedConnectionPoolLimits,
) -> Int {
  self.connection_capacity
}

///|
pub fn LunaOnlineFramedConnectionPoolLimits::io_poll_timeout_millis(
  self : LunaOnlineFramedConnectionPoolLimits,
) -> Int {
  self.io_poll_timeout_millis
}
