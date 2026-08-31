///|
/// Consumes the one already-prepared service into the existing OpenAI
/// Responses server exactly once. This qualification-only composition accepts
/// an opaque caller-admitted authentication capability and is restricted to
/// an explicitly configured loopback listener.
async fn RuntimeInstanceOwner::bind_openai_qualification(
  self : RuntimeInstanceOwner,
  policy : RuntimeOpenAiQualificationPolicy,
) -> Unit raise RuntimeInstanceError {
  self.bind_openai_owner(policy.server, OpenAiQualificationBinding)
}

///|
fn openai_qualification_ingress_shape_valid(
  server_count : Int,
  framed_pool_count : Int,
  openai_server_count : Int,
  openai_pool_count : Int,
  control_count : Int,
) -> Bool {
  server_count == 0 &&
  framed_pool_count == 0 &&
  openai_server_count + openai_pool_count == 1 &&
  control_count == 0
}

///|
fn openai_qualification_server_live_state_valid(
  state : @online_tcp.LunaOnlineTcpServerState,
  health : @online_tcp.LunaOnlineTcpServerHealth,
) -> Bool {
  health == LunaOnlineTcpServerHealthy &&
  state != LunaOnlineTcpServerDraining &&
  state != LunaOnlineTcpServerClosed
}

///|
fn openai_qualification_pool_live_state_valid(
  state : @online_tcp.LunaOnlineOpenAIConnectionPoolState,
  active_connections : Int,
  readiness : @online_tcp.LunaOnlineOpenAIConnectionPoolReadiness,
  health : @online_tcp.LunaOnlineTcpServerHealth,
) -> Bool {
  if active_connections < 0 ||
    readiness != LunaOnlineOpenAIConnectionPoolReady ||
    health != LunaOnlineTcpServerHealthy {
    return false
  }
  match state {
    LunaOnlineOpenAIConnectionPoolListening => active_connections == 0
    LunaOnlineOpenAIConnectionPoolActive => active_connections > 0
    LunaOnlineOpenAIConnectionPoolDraining
    | LunaOnlineOpenAIConnectionPoolClosed => false
  }
}

///|
fn RuntimeInstanceOwner::openai_qualification_ingress_shape_valid(
  self : RuntimeInstanceOwner,
) -> Bool {
  openai_qualification_ingress_shape_valid(
    self.servers.length(),
    self.framed_pools.length(),
    self.openai_servers.length(),
    self.openai_pools.length(),
    self.controls.length(),
  )
}

///|
fn RuntimeInstanceOwner::openai_qualification_ingress_live(
  self : RuntimeInstanceOwner,
) -> Bool {
  if !self.openai_qualification_ingress_shape_valid() {
    return false
  }
  if self.openai_servers.length() == 1 {
    return openai_qualification_server_live_state_valid(
      self.openai_servers[0].state(),
      self.openai_servers[0].health(),
    )
  }
  openai_qualification_pool_live_state_valid(
    self.openai_pools[0].state(),
    self.openai_pools[0].active_connection_count(),
    self.openai_pools[0].readiness(),
    self.openai_pools[0].health(),
  )
}

///|
fn RuntimeInstanceOwner::openai_qualification_ingress_active(
  self : RuntimeInstanceOwner,
) -> Bool {
  if !self.openai_qualification_ingress_live() {
    return false
  }
  if self.openai_servers.length() == 1 {
    return self.openai_servers[0].state() == LunaOnlineTcpServerConnected
  }
  self.openai_pools[0].state() == LunaOnlineOpenAIConnectionPoolActive &&
  self.openai_pools[0].active_connection_count() == 1
}

///|
fn RuntimeInstanceOwner::openai_qualification_ingress_idle(
  self : RuntimeInstanceOwner,
) -> Bool {
  if !self.openai_qualification_ingress_live() {
    return false
  }
  if self.openai_servers.length() == 1 {
    return self.openai_servers[0].state() == LunaOnlineTcpServerListening
  }
  self.openai_pools[0].state() == LunaOnlineOpenAIConnectionPoolListening &&
  self.openai_pools[0].active_connection_count() == 0
}

///|
/// Separate loopback qualification readiness; never production readiness.
pub fn RuntimeInstanceOwner::openai_qualification_readiness(
  self : RuntimeInstanceOwner,
) -> RuntimeOpenAiQualificationReadiness {
  if self.phase == OpenAiQualificationReady &&
    self.openai_qualification_ingress_live() {
    OpenAiQualificationListenerReady
  } else {
    OpenAiQualificationNotReady
  }
}
