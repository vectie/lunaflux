///|
/// Requires the exact post-child-Ready, pre-listener ownership state.
///
/// This proves only that the authenticated spawned child finished construction
/// of its physical executor and transferred one ready framed-service
/// preparation to this owner. It does not bind a listener, publish traffic
/// readiness, execute an inference plan, or attest numerical correctness.
pub fn RuntimeInstanceOwner::require_spawned_physical_handoff(
  self : RuntimeInstanceOwner,
) -> Unit raise RuntimeInstanceError {
  let preparation_ready = self.preparations.length() == 1 &&
    self.preparations[0].state() == LunaOnlineInstanceReady
  if !spawned_physical_handoff_matches(
      self.phase,
      preparation_ready,
      self.preparations.length(),
      self.cleanup.length(),
      self.services.length(),
      self.servers.length() + self.framed_pools.length(),
      self.openai_servers.length() + self.openai_pools.length(),
      self.borrowed_roots.length(),
    ) {
    raise fail(SpawnedPhysicalHandoff)
  }
}

///|
fn spawned_physical_handoff_matches(
  phase : RuntimeInstancePhase,
  preparation_ready : Bool,
  preparation_count : Int,
  cleanup_count : Int,
  service_count : Int,
  server_count : Int,
  openai_server_count : Int,
  borrowed_root_count : Int,
) -> Bool {
  phase == HealthyNotReady &&
  preparation_ready &&
  preparation_count == 1 &&
  cleanup_count == 0 &&
  service_count == 0 &&
  server_count == 0 &&
  openai_server_count == 0 &&
  borrowed_root_count == 0
}

///|
/// Rejects new admission through the existing server and begins owned drain.
pub fn RuntimeInstanceOwner::begin_drain(
  self : RuntimeInstanceOwner,
) -> Unit raise RuntimeInstanceError {
  match self.phase {
    Closed | Draining => return
    Ready | OpenAiQualificationReady => {
      let drain_failed = if self.servers.length() == 1 &&
        self.framed_pools.length() == 0 &&
        self.openai_servers.length() == 0 &&
        self.openai_pools.length() == 0 {
        try self.servers[0].begin_drain() catch {
          _ => true
        } noraise {
          _ => false
        }
      } else if self.framed_pools.length() == 1 &&
        self.servers.length() == 0 &&
        self.openai_servers.length() == 0 &&
        self.openai_pools.length() == 0 {
        self.framed_pools[0].begin_drain()
        false
      } else if self.openai_servers.length() == 1 &&
        self.servers.length() == 0 &&
        self.framed_pools.length() == 0 &&
        self.openai_pools.length() == 0 {
        try self.openai_servers[0].begin_drain() catch {
          _ => true
        } noraise {
          _ => false
        }
      } else if self.openai_pools.length() == 1 &&
        self.servers.length() == 0 &&
        self.framed_pools.length() == 0 &&
        self.openai_servers.length() == 0 {
        self.openai_pools[0].begin_drain()
        false
      } else {
        true
      }
      if drain_failed {
        self.phase = Failed
        raise fail(Lifecycle)
      }
      self.phase = Draining
    }
    HealthyNotReady => {
      if self.preparations.length() == 1 {
        let service = self.preparations[0].take_ready() catch {
          _ => raise fail(Lifecycle)
        }
        self.preparations.clear()
        service.begin_drain()
        self.services.push(service)
      }
      self.phase = Draining
    }
    Starting => self.phase = Draining
    Failed => ()
  }
}

///|
fn RuntimeInstanceOwner::progress_cleanup_service(
  self : RuntimeInstanceOwner,
) -> RuntimeInstanceProgress {
  if self.services.length() != 1 {
    return RuntimeFailedProgress
  }
  let readiness = self.services[0].progress() catch {
    _ => return RuntimeFailedProgress
  }
  if readiness == LunaOnlineFramedServiceClosed {
    self.services.clear()
    if self.phase == Draining {
      if self.close_inherited_drain(wait_for_response=true) {
        self.startup_binding.close_authentication()
        self.phase = Closed
        RuntimeClosedProgress
      } else {
        RuntimeDrainingProgress
      }
    } else {
      self.startup_binding.close_authentication()
      RuntimeFailedProgress
    }
  } else {
    RuntimeDrainingProgress
  }
}

///|
/// Advances at most one existing binder/server/service cleanup transition.
pub async fn RuntimeInstanceOwner::progress(
  self : RuntimeInstanceOwner,
) -> RuntimeInstanceProgress {
  if self.opaque_cli_requires_drain && !self.inherited_drain_required {
    self.phase = Failed
  }
  if !self.progress_inherited_drain() {
    self.phase = Failed
  }
  match self.phase {
    Starting => RuntimeStartingProgress
    HealthyNotReady => {
      if self.preparations.length() != 1 {
        self.phase = Failed
        return RuntimeFailedProgress
      }
      match self.startup_binding {
        RuntimeOpaqueOpenAiBinding(policy) => {
          self.bind_openai_owner(policy, OpenAiOpaqueCliProductionBinding) catch {
            _ => {
              self.phase = Failed
              return RuntimeFailedProgress
            }
          }
          return RuntimeReadyProgress
        }
        RuntimeOpaqueNativeBinding => ()
      }
      self.bind_native_ingress()
    }
    Ready =>
      if self.servers.length() == 1 &&
        self.framed_pools.length() == 0 &&
        self.openai_servers.length() == 0 &&
        self.openai_pools.length() == 0 {
        let ingress = self.servers[0].progress_on_reactor()
        if self.servers[0].health() == LunaOnlineTcpServerFailed {
          self.phase = Failed
          return RuntimeFailedProgress
        }
        if self.servers[0].state() == LunaOnlineTcpServerDraining ||
          self.servers[0].state() == LunaOnlineTcpServerClosed {
          self.phase = Failed
          return RuntimeFailedProgress
        }
        if !self.progress_control_after_ingress(
            ingress == LunaOnlineTcpServerBackpressured ||
            ingress == LunaOnlineTcpServerAcceptTimedOut,
          ) {
          self.phase = Failed
          return RuntimeFailedProgress
        }
        RuntimeServingProgress
      } else if self.framed_pools.length() == 1 &&
        self.servers.length() == 0 &&
        self.openai_servers.length() == 0 &&
        self.openai_pools.length() == 0 {
        let ingress = self.framed_pools[0].progress_on_reactor()
        if self.framed_pools[0].health() == LunaOnlineTcpServerFailed ||
          self.framed_pools[0].state() == LunaOnlineFramedConnectionPoolDraining ||
          self.framed_pools[0].state() == LunaOnlineFramedConnectionPoolClosed {
          self.phase = Failed
          return RuntimeFailedProgress
        }
        if !self.progress_control_after_ingress(
            ingress == LunaOnlineFramedConnectionPoolBackpressured,
          ) {
          self.phase = Failed
          return RuntimeFailedProgress
        }
        RuntimeServingProgress
      } else if self.openai_servers.length() == 1 &&
        self.servers.length() == 0 &&
        self.framed_pools.length() == 0 &&
        self.openai_pools.length() == 0 {
        let ingress = self.openai_servers[0].progress_on_reactor()
        if self.openai_servers[0].health() == LunaOnlineTcpServerFailed {
          self.phase = Failed
          return RuntimeFailedProgress
        }
        if self.openai_servers[0].state() == LunaOnlineTcpServerDraining ||
          self.openai_servers[0].state() == LunaOnlineTcpServerClosed {
          self.phase = Failed
          return RuntimeFailedProgress
        }
        if !self.progress_control_after_ingress(
            ingress == LunaOnlineTcpServerBackpressured ||
            ingress == LunaOnlineTcpServerAcceptTimedOut,
          ) {
          self.phase = Failed
          return RuntimeFailedProgress
        }
        RuntimeServingProgress
      } else if self.openai_pools.length() == 1 &&
        self.servers.length() == 0 &&
        self.framed_pools.length() == 0 &&
        self.openai_servers.length() == 0 {
        let ingress = self.openai_pools[0].progress_on_reactor()
        if self.openai_pools[0].health() == LunaOnlineTcpServerFailed ||
          self.openai_pools[0].state() == LunaOnlineOpenAIConnectionPoolDraining ||
          self.openai_pools[0].state() == LunaOnlineOpenAIConnectionPoolClosed {
          self.phase = Failed
          return RuntimeFailedProgress
        }
        if !self.progress_control_after_ingress(
            ingress == LunaOnlineOpenAIConnectionPoolBackpressured,
          ) {
          self.phase = Failed
          return RuntimeFailedProgress
        }
        RuntimeServingProgress
      } else {
        self.phase = Failed
        RuntimeFailedProgress
      }
    OpenAiQualificationReady =>
      if self.openai_servers.length() == 1 &&
        self.openai_pools.length() == 0 &&
        self.servers.length() == 0 &&
        self.framed_pools.length() == 0 {
        ignore(self.openai_servers[0].progress_on_reactor())
        if self.openai_servers[0].health() == LunaOnlineTcpServerFailed {
          self.phase = Failed
          return RuntimeFailedProgress
        }
        if self.openai_servers[0].state() == LunaOnlineTcpServerDraining ||
          self.openai_servers[0].state() == LunaOnlineTcpServerClosed {
          self.phase = Failed
          return RuntimeFailedProgress
        }
        if !self.progress_control() {
          self.phase = Failed
          return RuntimeFailedProgress
        }
        RuntimeServingProgress
      } else if self.openai_pools.length() == 1 &&
        self.openai_servers.length() == 0 &&
        self.servers.length() == 0 &&
        self.framed_pools.length() == 0 {
        ignore(self.openai_pools[0].progress_on_reactor())
        if self.openai_pools[0].health() == LunaOnlineTcpServerFailed ||
          self.openai_pools[0].state() == LunaOnlineOpenAIConnectionPoolDraining ||
          self.openai_pools[0].state() == LunaOnlineOpenAIConnectionPoolClosed {
          self.phase = Failed
          return RuntimeFailedProgress
        }
        if !self.progress_control() {
          self.phase = Failed
          return RuntimeFailedProgress
        }
        RuntimeServingProgress
      } else {
        self.phase = Failed
        RuntimeFailedProgress
      }
    Draining =>
      if self.servers.length() == 1 {
        ignore(self.servers[0].progress_on_reactor())
        if self.servers[0].state() == LunaOnlineTcpServerClosed {
          self.servers.clear()
          if self.close_control() &&
            self.close_inherited_drain(wait_for_response=true) {
            self.startup_binding.close_authentication()
            self.phase = Closed
            RuntimeClosedProgress
          } else {
            RuntimeDrainingProgress
          }
        } else {
          if self.controls.length() == 1 &&
            self.controls[0].state() != LunaOnlineHttpControlFailed {
            ignore(self.progress_control())
          }
          RuntimeDrainingProgress
        }
      } else if self.framed_pools.length() == 1 {
        ignore(self.framed_pools[0].progress_on_reactor())
        if self.framed_pools[0].state() == LunaOnlineFramedConnectionPoolClosed {
          self.framed_pools.clear()
          if self.close_control() &&
            self.close_inherited_drain(wait_for_response=true) {
            self.startup_binding.close_authentication()
            self.phase = Closed
            RuntimeClosedProgress
          } else {
            RuntimeDrainingProgress
          }
        } else {
          if self.controls.length() == 1 &&
            self.controls[0].state() != LunaOnlineHttpControlFailed {
            ignore(self.progress_control())
          }
          RuntimeDrainingProgress
        }
      } else if self.openai_servers.length() == 1 {
        ignore(self.openai_servers[0].progress_on_reactor())
        if self.openai_servers[0].state() == LunaOnlineTcpServerClosed {
          self.openai_servers.clear()
          if self.close_control() &&
            self.close_inherited_drain(wait_for_response=true) {
            self.startup_binding.close_authentication()
            self.phase = Closed
            RuntimeClosedProgress
          } else {
            RuntimeDrainingProgress
          }
        } else {
          if self.controls.length() == 1 &&
            self.controls[0].state() != LunaOnlineHttpControlFailed {
            ignore(self.progress_control())
          }
          RuntimeDrainingProgress
        }
      } else if self.openai_pools.length() == 1 {
        ignore(self.openai_pools[0].progress_on_reactor())
        if self.openai_pools[0].state() == LunaOnlineOpenAIConnectionPoolClosed {
          self.openai_pools.clear()
          if self.close_control() &&
            self.close_inherited_drain(wait_for_response=true) {
            self.startup_binding.close_authentication()
            self.phase = Closed
            RuntimeClosedProgress
          } else {
            RuntimeDrainingProgress
          }
        } else {
          if self.controls.length() == 1 &&
            self.controls[0].state() != LunaOnlineHttpControlFailed {
            ignore(self.progress_control())
          }
          RuntimeDrainingProgress
        }
      } else if self.controls.length() == 1 {
        if self.close_control() &&
          self.close_inherited_drain(wait_for_response=true) {
          self.startup_binding.close_authentication()
          self.phase = Closed
          RuntimeClosedProgress
        } else {
          RuntimeDrainingProgress
        }
      } else if self.services.length() == 0 &&
        self.preparations.length() == 0 &&
        self.cleanup.length() == 0 {
        if self.close_inherited_drain(wait_for_response=true) {
          self.startup_binding.close_authentication()
          self.phase = Closed
          RuntimeClosedProgress
        } else {
          RuntimeDrainingProgress
        }
      } else {
        self.progress_cleanup_service()
      }
    Failed => {
      if self.servers.length() == 1 {
        self.servers[0].begin_drain() catch {
          _ => return RuntimeFailedProgress
        }
        ignore(self.servers[0].progress_on_reactor())
        if self.servers[0].state() == LunaOnlineTcpServerClosed {
          self.servers.clear()
        }
      } else if self.framed_pools.length() == 1 {
        self.framed_pools[0].begin_drain()
        ignore(self.framed_pools[0].progress_on_reactor())
        if self.framed_pools[0].state() == LunaOnlineFramedConnectionPoolClosed {
          self.framed_pools.clear()
        }
      } else if self.openai_servers.length() == 1 {
        self.openai_servers[0].begin_drain() catch {
          _ => return RuntimeFailedProgress
        }
        ignore(self.openai_servers[0].progress_on_reactor())
        if self.openai_servers[0].state() == LunaOnlineTcpServerClosed {
          self.openai_servers.clear()
        }
      } else if self.openai_pools.length() == 1 {
        self.openai_pools[0].begin_drain()
        ignore(self.openai_pools[0].progress_on_reactor())
        if self.openai_pools[0].state() == LunaOnlineOpenAIConnectionPoolClosed {
          self.openai_pools.clear()
        }
      } else if self.controls.length() == 1 {
        ignore(self.close_control())
      } else if self.services.length() == 1 {
        ignore(self.progress_cleanup_service())
      } else if self.preparations.length() == 1 {
        match self.preparations[0].state() {
          LunaOnlineInstanceReady => {
            let service = self.preparations[0].take_ready() catch {
              _ => return RuntimeFailedProgress
            }
            self.preparations.clear()
            service.begin_drain()
            self.services.push(service)
          }
          LunaOnlineInstanceCleanupRequired => {
            let cleanup = self.preparations[0].take_cleanup() catch {
              _ => return RuntimeFailedProgress
            }
            self.preparations.clear()
            self.cleanup.push(cleanup)
          }
        }
      } else if self.cleanup.length() == 1 {
        self.cleanup[0].retry_close() catch {
          _ => return RuntimeFailedProgress
        }
        if self.cleanup[0].is_closed() {
          self.cleanup.clear()
        }
      } else if self.borrowed_roots.length() != 0 {
        let index = self.borrowed_roots.length() - 1
        let root = self.borrowed_roots[index]
        root.close() catch {
          _ => return RuntimeFailedProgress
        }
        if root.is_closed() {
          ignore(self.borrowed_roots.pop())
        }
      } else if self.inherited_drains.length() != 0 {
        ignore(self.close_inherited_drain(wait_for_response=false))
      } else {
        self.startup_binding.close_authentication()
      }
      RuntimeFailedProgress
    }
    Closed => RuntimeClosedProgress
  }
}

///|
/// True only after every lower cleanup authority has terminally closed.
pub fn RuntimeInstanceOwner::cleanup_complete(
  self : RuntimeInstanceOwner,
) -> Bool {
  self.preparations.length() == 0 &&
  self.cleanup.length() == 0 &&
  self.services.length() == 0 &&
  self.servers.length() == 0 &&
  self.framed_pools.length() == 0 &&
  self.openai_servers.length() == 0 &&
  self.openai_pools.length() == 0 &&
  self.controls.length() == 0 &&
  self.inherited_drains.length() == 0 &&
  self.borrowed_roots.length() == 0
}
