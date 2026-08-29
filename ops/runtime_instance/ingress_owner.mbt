///|
fn runtime_uses_concurrent_ingress(connection_capacity : Int) -> Bool {
  connection_capacity > 1
}

///|
fn RuntimeInstanceOwner::record_cold_start(self : RuntimeInstanceOwner) -> Bool {
  if self.cold_start_recorded ||
    self.servers.length() +
    self.framed_pools.length() +
    self.openai_servers.length() +
    self.openai_pools.length() !=
    1 ||
    (
      self.servers.length() + self.framed_pools.length() == 1 &&
      self.controls.length() != 1
    ) ||
    (
      self.openai_servers.length() + self.openai_pools.length() == 1 &&
      self.controls.length() != 0
    ) {
    return false
  }
  let now = self.clock.now_millis() catch { _ => return false }
  if now < self.started_at_millis || now - self.started_at_millis > 86400000UL {
    return false
  }
  let duration_millis = (now - self.started_at_millis).to_int()
  if self.servers.length() == 1 {
    self.servers[0].record_cold_start_latency_millis(duration_millis) catch {
      _ => return false
    }
  } else if self.framed_pools.length() == 1 {
    self.framed_pools[0].record_cold_start_latency_millis(duration_millis) catch {
      _ => return false
    }
  } else if self.openai_servers.length() == 1 {
    self.openai_servers[0].record_cold_start_latency_millis(duration_millis) catch {
      _ => return false
    }
  } else {
    self.openai_pools[0].record_cold_start_latency_millis(duration_millis) catch {
      _ => return false
    }
  }
  if self.controls.length() == 1 {
    self.controls[0].publish_startup_measurement(duration_millis) catch {
      _ => return false
    }
  }
  self.cold_start_latency_millis = duration_millis
  self.cold_start_recorded = true
  true
}

///|
async fn RuntimeInstanceOwner::bind_native_ingress(
  self : RuntimeInstanceOwner,
) -> RuntimeInstanceProgress {
  let preparation = self.preparations[0]
  if !runtime_uses_concurrent_ingress(self.connection_capacity) {
    let server = @online_tcp.bind_luna_online_tcp_pipeline_server(
      self.addr,
      preparation,
      self.transport,
      log_capacity=self.telemetry_capacity,
    ) catch {
      _ => {
        let service = preparation.take_ready() catch {
          _ => {
            self.phase = Failed
            return RuntimeFailedProgress
          }
        }
        self.preparations.clear()
        service.begin_drain()
        self.services.push(service)
        self.phase = Failed
        return RuntimeFailedProgress
      }
    }
    self.preparations.clear()
    self.servers.push(server)
    let control = bind_native_runtime_control(server, self.transport) catch {
      _ => {
        server.begin_drain() catch {
          _ => ()
        }
        self.phase = Failed
        return RuntimeFailedProgress
      }
    }
    self.controls.push(control)
    if !self.record_cold_start() {
      server.begin_drain() catch {
        _ => ()
      }
      self.phase = Failed
      return RuntimeFailedProgress
    }
  } else {
    let pool_limits = @online_tcp.LunaOnlineFramedConnectionPoolLimits::new(
      connection_capacity=self.connection_capacity,
      io_poll_timeout_millis=self.transport.pool_poll_timeout_millis(),
      request_work_units=self.request_work_units,
      event_work_units=self.event_work_units,
    ) catch {
      _ => {
        self.phase = Failed
        return RuntimeFailedProgress
      }
    }
    let prepared = @online_tcp.prepare_luna_online_framed_connection_pool(
      preparation,
      self.transport,
      pool_limits,
      self.receipt_deadline,
    ) catch {
      _ => {
        self.phase = Failed
        return RuntimeFailedProgress
      }
    }
    let pool = @online_tcp.bind_luna_online_framed_connection_pool(
      self.addr,
      prepared,
      self.transport,
      log_capacity=self.telemetry_capacity,
    ) catch {
      _ => {
        self.phase = Failed
        return RuntimeFailedProgress
      }
    }
    self.preparations.clear()
    self.framed_pools.push(pool)
    let control = bind_native_pool_runtime_control(pool, self.transport) catch {
      _ => {
        pool.begin_drain()
        self.phase = Failed
        return RuntimeFailedProgress
      }
    }
    self.controls.push(control)
    if !self.record_cold_start() {
      pool.begin_drain()
      self.phase = Failed
      return RuntimeFailedProgress
    }
  }
  self.phase = Ready
  RuntimeReadyProgress
}
