///|
const RUNTIME_CONTROL_MAX_HEAD_BYTES : Int = 1024

///|
const RUNTIME_CONTROL_FAIRNESS_TRANSITIONS : Int = 64

///|
fn runtime_control_limits() -> @http1.LunaHttp1ControlLimits raise RuntimeInstanceError {
  @http1.LunaHttp1ControlLimits::new(
    max_head_bytes=RUNTIME_CONTROL_MAX_HEAD_BYTES,
  ) catch {
    _ => raise fail(ListenerBinding)
  }
}

///|
async fn bind_native_runtime_control(
  server : @online_tcp.LunaOnlineTcpServer,
  transport : @online_tcp.LunaOnlineTcpLimits,
) -> @online_tcp.LunaOnlineHttpControlServer {
  let preparation = @online_tcp.prepare_luna_online_http_control_for_native(
    server,
    runtime_control_limits(),
  ) catch {
    _ => raise fail(ListenerBinding)
  }
  @online_tcp.bind_luna_online_http_control_server(preparation, transport) catch {
    _ => raise fail(ListenerBinding)
  }
}

///|
async fn bind_native_pool_runtime_control(
  pool : @online_tcp.LunaOnlineFramedConnectionPool,
  transport : @online_tcp.LunaOnlineTcpLimits,
) -> @online_tcp.LunaOnlineHttpControlServer {
  let preparation = @online_tcp.prepare_luna_online_http_control_for_native_pool(
    pool,
    runtime_control_limits(),
  ) catch {
    _ => raise fail(ListenerBinding)
  }
  @online_tcp.bind_luna_online_http_control_server(preparation, transport) catch {
    _ => raise fail(ListenerBinding)
  }
}

///|
fn RuntimeInstanceOwner::operational_local_addr(
  self : RuntimeInstanceOwner,
) -> @socket.Addr raise RuntimeInstanceError {
  if self.phase == OpenAiQualificationReady &&
    self.controls.length() == 0 &&
    self.openai_servers.length() + self.openai_pools.length() == 1 {
    return self.singular_ingress_addr()
  }
  if self.phase == Failed ||
    self.phase == Closed ||
    self.controls.length() != 1 ||
    self.controls[0].state() == LunaOnlineHttpControlClosed ||
    self.controls[0].state() == LunaOnlineHttpControlFailed {
    raise fail(ListenerBinding)
  }
  self.controls[0].local_addr()
}

///|
async fn RuntimeInstanceOwner::progress_control(
  self : RuntimeInstanceOwner,
) -> Bool {
  if self.controls.length() == 0 &&
    self.openai_servers.length() + self.openai_pools.length() == 1 {
    return true
  }
  if self.controls.length() != 1 {
    return false
  }
  let progress = self.controls[0].progress_on_reactor()
  progress != LunaOnlineHttpControlFailedProgress &&
  self.controls[0].state() != LunaOnlineHttpControlFailed
}

///|
async fn RuntimeInstanceOwner::progress_control_after_ingress(
  self : RuntimeInstanceOwner,
  ingress_backpressured : Bool,
) -> Bool {
  if self.control_deferred_transitions < RUNTIME_CONTROL_FAIRNESS_TRANSITIONS {
    self.control_deferred_transitions += 1
  }
  if !ingress_backpressured &&
    self.control_deferred_transitions < RUNTIME_CONTROL_FAIRNESS_TRANSITIONS {
    return true
  }
  self.control_deferred_transitions = 0
  self.progress_control()
}

///|
async fn RuntimeInstanceOwner::close_control(
  self : RuntimeInstanceOwner,
) -> Bool {
  if self.controls.length() == 0 {
    return true
  }
  if self.controls.length() != 1 {
    return false
  }
  self.controls[0].begin_drain()
  ignore(self.controls[0].progress_on_reactor())
  if self.controls[0].state() == LunaOnlineHttpControlClosed {
    self.controls.clear()
    true
  } else {
    false
  }
}
