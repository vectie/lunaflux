///|
fn luna_online_tcp_server_health(
  failure_rule : Int,
  closed : Bool,
) -> LunaOnlineTcpServerHealth {
  if failure_rule == 0 && !closed {
    LunaOnlineTcpServerHealthy
  } else {
    LunaOnlineTcpServerFailed
  }
}

///|
/// Reports sticky lower service/worker/transport failure without payloads.
pub fn LunaOnlineTcpServer::health(
  self : LunaOnlineTcpServer,
) -> LunaOnlineTcpServerHealth {
  luna_online_tcp_server_health(
    self.failure_rule,
    self.phase == LUNA_ONLINE_TCP_SERVER_CLOSED,
  )
}

///|
/// Reports sticky lower service/worker/transport failure for the OpenAI server
/// without exposing protocol or payload detail.
pub fn LunaOnlineOpenAIServer::health(
  self : LunaOnlineOpenAIServer,
) -> LunaOnlineTcpServerHealth {
  luna_online_tcp_server_health(
    self.failure_rule,
    self.phase == LUNA_ONLINE_TCP_SERVER_CLOSED,
  )
}
