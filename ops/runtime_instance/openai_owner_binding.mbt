///|
/// Private publication provenance. Only the production constructor can carry
/// the explicit ingress approval required to publish ordinary readiness.
priv enum RuntimeOpenAiBindingAuthority {
  OpenAiQualificationBinding
  OpenAiEmbeddingProductionBinding(RuntimeOpenAiLoopbackIngressApproval)
  OpenAiOpaqueCliProductionBinding
}

///|
fn RuntimeOpenAiBindingAuthority::failure_stage(
  self : RuntimeOpenAiBindingAuthority,
) -> RuntimeInstanceStage {
  match self {
    OpenAiQualificationBinding => OpenAiQualification
    OpenAiEmbeddingProductionBinding(_) => OpenAiProduction
    OpenAiOpaqueCliProductionBinding => OpenAiProduction
  }
}

///|
fn openai_binding_can_publish_ready(
  authority : RuntimeOpenAiBindingAuthority,
  loopback_allowed : Bool,
  listener_allowed : Bool,
) -> Bool {
  match authority {
    OpenAiQualificationBinding => false
    OpenAiEmbeddingProductionBinding(ingress) =>
      loopback_allowed && ingress.is_valid()
    OpenAiOpaqueCliProductionBinding => listener_allowed
  }
}

///|
fn openai_binding_state_valid(
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
/// Consumes the singular ready lower service into exactly one OpenAI server.
/// Qualification and embedding-production authority share mechanics but have
/// distinct publication provenance and failure stages.
async fn RuntimeInstanceOwner::bind_openai_owner(
  self : RuntimeInstanceOwner,
  policy : RuntimeOpenAiServerPolicy,
  authority : RuntimeOpenAiBindingAuthority,
) -> Unit raise RuntimeInstanceError {
  let stage = authority.failure_stage()
  let transferred = Ref(false)
  defer (if !transferred.val { policy.auth_policy.close() })
  let preparation_ready = self.preparations.length() == 1 &&
    self.preparations[0].state() == LunaOnlineInstanceReady
  if !self.openai_listener_allowed ||
    !openai_binding_state_valid(
      self.phase,
      preparation_ready,
      self.preparations.length(),
      self.cleanup.length(),
      self.services.length(),
      self.servers.length() + self.framed_pools.length(),
      self.openai_servers.length() + self.openai_pools.length(),
      self.borrowed_roots.length(),
    ) {
    raise fail(stage)
  }
  match authority {
    OpenAiQualificationBinding if !self.loopback_openai_allowed =>
      raise fail(stage)
    OpenAiEmbeddingProductionBinding(ingress) if !ingress.is_valid() =>
      raise fail(stage)
    OpenAiEmbeddingProductionBinding(_) if !self.loopback_openai_allowed =>
      raise fail(stage)
    _ => ()
  }
  let service = self.preparations[0]
  let publish_ready = openai_binding_can_publish_ready(
    authority,
    self.loopback_openai_allowed,
    self.openai_listener_allowed,
  )
  if !runtime_uses_concurrent_ingress(self.connection_capacity) {
    let prepared = @online_tcp.prepare_luna_online_openai_server(
      service,
      policy.http_limits,
      policy.http_budget,
      policy.response_budget,
      policy.auth_policy,
      policy.inbound_budget,
      policy.outbound_budget,
      policy.inbound_limits,
      policy.template,
      policy.model_alias,
      model_alias_offset=0,
      model_alias_length=policy.model_alias.length(),
      policy.response_id_prefix,
      response_id_prefix_offset=0,
      response_id_prefix_length=policy.response_id_prefix.length(),
      policy.cache_scope,
      max_new_tokens=policy.max_new_tokens,
      context_ceiling=policy.context_ceiling,
      policy.sampling,
      policy.sampling_seed,
      policy.deadline,
      production_ready=publish_ready,
    ) catch {
      _ => raise fail(stage)
    }
    let server = @online_tcp.bind_luna_online_openai_server(
      self.addr,
      prepared,
      self.transport,
      log_capacity=self.telemetry_capacity,
    ) catch {
      _ => {
        self.phase = Failed
        raise fail(stage)
      }
    }
    transferred.val = true
    self.openai_servers.push(server)
  } else {
    let pool_limits = @online_tcp.LunaOnlineOpenAIConnectionPoolLimits::new(
      connection_capacity=self.connection_capacity,
      output_buffer_bytes=self.transport.write_chunk_bytes(),
      io_poll_timeout_millis=self.transport.pool_poll_timeout_millis(),
    ) catch {
      _ => raise fail(stage)
    }
    let prepared = @online_tcp.prepare_luna_online_openai_connection_pool(
      service,
      self.transport,
      pool_limits,
      policy.http_limits,
      policy.http_budget,
      policy.response_budget,
      policy.auth_policy,
      policy.inbound_budget,
      policy.outbound_budget,
      policy.inbound_limits,
      policy.template,
      policy.model_alias,
      model_alias_offset=0,
      model_alias_length=policy.model_alias.length(),
      policy.response_id_prefix,
      response_id_prefix_offset=0,
      response_id_prefix_length=policy.response_id_prefix.length(),
      policy.cache_scope,
      max_new_tokens=policy.max_new_tokens,
      context_ceiling=policy.context_ceiling,
      policy.sampling,
      policy.sampling_seed,
      policy.deadline,
      production_ready=publish_ready,
    ) catch {
      _ => raise fail(stage)
    }
    let pool = @online_tcp.bind_luna_online_openai_connection_pool(
      self.addr,
      prepared,
      self.transport,
      log_capacity=self.telemetry_capacity,
    ) catch {
      _ => {
        self.phase = Failed
        raise fail(stage)
      }
    }
    transferred.val = true
    self.openai_pools.push(pool)
  }
  self.preparations.clear()
  if !self.record_cold_start() {
    if self.openai_servers.length() == 1 {
      self.openai_servers[0].begin_drain() catch {
        _ => ()
      }
    } else if self.openai_pools.length() == 1 {
      self.openai_pools[0].begin_drain()
    }
    self.phase = Failed
    raise fail(ColdStartTelemetry)
  }
  self.phase = if publish_ready { Ready } else { OpenAiQualificationReady }
}
