///|
fn runtime_bound_endpoint_publication_allowed(
  phase : RuntimeInstancePhase,
  health : RuntimeInstanceHealth,
  readiness : RuntimeInstanceReadiness,
) -> Bool {
  phase == Ready && health == RuntimeHealthy && readiness == RuntimeReady
}

///|
fn RuntimeInstanceOwner::singular_ingress_addr(
  self : RuntimeInstanceOwner,
) -> @socket.Addr raise RuntimeInstanceError {
  let count = self.servers.length() +
    self.framed_pools.length() +
    self.openai_servers.length() +
    self.openai_pools.length()
  if count != 1 {
    raise fail(Lifecycle)
  }
  if self.servers.length() == 1 {
    return self.servers[0].local_addr()
  }
  if self.framed_pools.length() == 1 {
    return self.framed_pools[0].local_addr()
  }
  if self.openai_servers.length() == 1 {
    return self.openai_servers[0].local_addr()
  }
  guard self.openai_pools.length() == 1 else { raise fail(Lifecycle) }
  self.openai_pools[0].local_addr()
}

///|
fn runtime_bound_endpoint_from_addr(
  protocol : RuntimeServiceProtocol,
  addr : @socket.Addr,
) -> RuntimeBoundEndpoint raise RuntimeInstanceError {
  if addr.port() <= 0 || addr.port() > 65535 {
    raise fail(Lifecycle)
  }
  let scheme = match protocol {
    RuntimeNativeFramedV1 => "tcp://"
    RuntimeOpenAiResponsesV1 => "http://"
  }
  { protocol, origin: scheme + addr.to_string() }
}

///|
fn runtime_control_origin_publication_allowed(
  phase : RuntimeInstancePhase,
  health : RuntimeInstanceHealth,
  control_count : Int,
  control_state : @online_tcp.LunaOnlineHttpControlState,
  control_health : @online_tcp.LunaOnlineTcpServerHealth,
) -> Bool {
  phase == Ready &&
  health == RuntimeHealthy &&
  control_count == 1 &&
  (
    control_state == LunaOnlineHttpControlListening ||
    control_state == LunaOnlineHttpControlConnected
  ) &&
  control_health == LunaOnlineTcpServerHealthy
}

///|
fn runtime_control_origin_from_addr(
  addr : @socket.Addr,
) -> String raise RuntimeInstanceError {
  if addr.port() <= 0 || addr.port() > 65535 {
    raise fail(Lifecycle)
  }
  "http://" + addr.to_string()
}

///|
/// Publishes the actual singular ingress only after health and traffic
/// readiness agree. Qualification, startup, drain, failure, and closed states
/// cannot expose a stale or internal control origin.
pub fn RuntimeInstanceOwner::bound_endpoint(
  self : RuntimeInstanceOwner,
) -> RuntimeBoundEndpoint raise RuntimeInstanceError {
  if !runtime_bound_endpoint_publication_allowed(
      self.phase,
      self.health(),
      self.readiness(),
    ) {
    raise fail(Lifecycle)
  }
  let addr = self.singular_ingress_addr()
  runtime_bound_endpoint_from_addr(self.service_protocol(), addr)
}

///|
/// Publishes the root-free observational origin only while the authenticated
/// owner is live and healthy. OpenAI shares its serving origin; native-framed
/// retains the separate loopback-only HTTP control listener.
pub fn RuntimeInstanceOwner::control_origin(
  self : RuntimeInstanceOwner,
) -> String raise RuntimeInstanceError {
  if self.service_protocol() == RuntimeOpenAiResponsesV1 {
    return self.bound_endpoint().origin()
  }
  if self.controls.length() != 1 {
    raise fail(Lifecycle)
  }
  let control = self.controls[0]
  if !runtime_control_origin_publication_allowed(
      self.phase,
      self.health(),
      self.controls.length(),
      control.state(),
      control.health(),
    ) {
    raise fail(Lifecycle)
  }
  runtime_control_origin_from_addr(control.local_addr())
}
