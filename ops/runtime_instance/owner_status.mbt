///|
pub fn RuntimeInstanceOwner::phase(
  self : RuntimeInstanceOwner,
) -> RuntimeInstancePhase {
  self.phase
}

///|
pub fn RuntimeInstanceOwner::service_protocol(
  self : RuntimeInstanceOwner,
) -> RuntimeServiceProtocol {
  match self.startup_binding {
    RuntimeOpaqueNativeBinding => RuntimeNativeFramedV1
    RuntimeOpaqueOpenAiBinding(_) => RuntimeOpenAiResponsesV1
  }
}

///|
pub fn RuntimeInstanceOwner::health(
  self : RuntimeInstanceOwner,
) -> RuntimeInstanceHealth {
  if self.phase == Ready && !self.ready_server_healthy() {
    return RuntimeUnhealthy
  }
  health_for_phase(self.phase)
}

///|
fn health_for_phase(phase : RuntimeInstancePhase) -> RuntimeInstanceHealth {
  match phase {
    Starting => RuntimeStarting
    HealthyNotReady | Ready | OpenAiQualificationReady | Draining =>
      RuntimeHealthy
    Failed => RuntimeUnhealthy
    Closed => RuntimeClosed
  }
}

///|
pub fn RuntimeInstanceOwner::readiness(
  self : RuntimeInstanceOwner,
) -> RuntimeInstanceReadiness {
  readiness_for_phase(self.phase, self.ready_server_ready())
}

///|
pub fn RuntimeStartupMeasurement::cold_start_latency_millis(
  self : RuntimeStartupMeasurement,
) -> Int {
  self.cold_start_latency_millis
}

///|
pub fn RuntimeStartupMeasurement::readiness(
  self : RuntimeStartupMeasurement,
) -> RuntimeInstanceReadiness {
  self.readiness
}

///|
/// Returns the exact startup duration only after its metrics publication has
/// succeeded. Readiness is re-projected from the live owner lifecycle so drain,
/// failure, and qualification states can never retain a stale true value.
pub fn RuntimeInstanceOwner::startup_measurement(
  self : RuntimeInstanceOwner,
) -> RuntimeStartupMeasurement raise RuntimeInstanceError {
  if !self.cold_start_recorded ||
    self.cold_start_latency_millis < 0 ||
    self.cold_start_latency_millis > 86400000 {
    raise fail(ColdStartTelemetry)
  }
  {
    cold_start_latency_millis: self.cold_start_latency_millis,
    readiness: self.readiness(),
  }
}

///|
fn RuntimeInstanceOwner::ready_server_healthy(
  self : RuntimeInstanceOwner,
) -> Bool {
  let native_control_healthy = self.controls.length() == 1 &&
    self.controls[0].health() == LunaOnlineTcpServerHealthy
  let openai_control_healthy = self.controls.length() == 0
  let native_healthy = self.servers.length() == 1 &&
    self.framed_pools.length() == 0 &&
    self.openai_servers.length() == 0 &&
    self.openai_pools.length() == 0 &&
    self.servers[0].state() != LunaOnlineTcpServerDraining &&
    self.servers[0].state() != LunaOnlineTcpServerClosed &&
    self.servers[0].health() == LunaOnlineTcpServerHealthy
  let openai_healthy = self.openai_servers.length() == 1 &&
    self.servers.length() == 0 &&
    self.framed_pools.length() == 0 &&
    self.openai_pools.length() == 0 &&
    self.openai_servers[0].state() != LunaOnlineTcpServerDraining &&
    self.openai_servers[0].state() != LunaOnlineTcpServerClosed &&
    self.openai_servers[0].health() == LunaOnlineTcpServerHealthy
  let native_pool_healthy = self.framed_pools.length() == 1 &&
    self.servers.length() == 0 &&
    self.openai_servers.length() == 0 &&
    self.openai_pools.length() == 0 &&
    self.framed_pools[0].health() == LunaOnlineTcpServerHealthy &&
    self.framed_pools[0].readiness() == LunaOnlineFramedConnectionPoolReady
  let openai_pool_healthy = self.openai_pools.length() == 1 &&
    self.servers.length() == 0 &&
    self.framed_pools.length() == 0 &&
    self.openai_servers.length() == 0 &&
    self.openai_pools[0].health() == LunaOnlineTcpServerHealthy &&
    self.openai_pools[0].readiness() == LunaOnlineOpenAIConnectionPoolReady
  runtime_live_health(
    (native_control_healthy && (native_healthy || native_pool_healthy)) ||
    (openai_control_healthy && (openai_healthy || openai_pool_healthy)),
    true,
    self.inherited_drain_live(),
  )
}

///|
fn RuntimeInstanceOwner::ready_server_ready(
  self : RuntimeInstanceOwner,
) -> Bool {
  if self.openai_servers.length() + self.openai_pools.length() == 1 {
    self.ready_server_healthy()
  } else {
    self.ready_server_healthy() &&
    self.controls.length() == 1 &&
    self.controls[0].readiness() == LunaOnlineHttpControlReady
  }
}

///|
fn runtime_live_health(
  control_healthy : Bool,
  execution_healthy : Bool,
  drain_authority_live : Bool,
) -> Bool {
  control_healthy && execution_healthy && drain_authority_live
}

///|
/// Returns the existing bounded instance metrics snapshot when server-owned.
fn RuntimeInstanceOwner::metrics_snapshot(
  self : RuntimeInstanceOwner,
) -> @metrics.LunaInstanceMetricsSnapshot raise RuntimeInstanceError {
  if self.servers.length() == 1 && self.openai_servers.length() == 0 {
    return self.servers[0].metrics_snapshot() catch {
      _ => raise fail(Lifecycle)
    }
  }
  if self.framed_pools.length() == 1 &&
    self.servers.length() == 0 &&
    self.openai_servers.length() == 0 &&
    self.openai_pools.length() == 0 {
    return self.framed_pools[0].metrics_snapshot() catch {
      _ => raise fail(Lifecycle)
    }
  }
  if self.openai_servers.length() == 1 && self.servers.length() == 0 {
    return self.openai_servers[0].metrics_snapshot() catch {
      _ => raise fail(Lifecycle)
    }
  }
  if self.openai_pools.length() == 1 &&
    self.servers.length() == 0 &&
    self.framed_pools.length() == 0 &&
    self.openai_servers.length() == 0 {
    return self.openai_pools[0].metrics_snapshot() catch {
      _ => raise fail(Lifecycle)
    }
  }
  raise fail(Lifecycle)
}

///|
fn readiness_for_phase(
  phase : RuntimeInstancePhase,
  listener_bound : Bool,
) -> RuntimeInstanceReadiness {
  if phase == Ready && listener_bound {
    RuntimeReady
  } else {
    RuntimeNotReady
  }
}
