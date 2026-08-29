///|
fn reject(
  stage : RuntimeInstanceAdmissionStage,
) -> RuntimeInstanceAdmissionError {
  Rejected(stage)
}

///|
fn admit_tokenizer_binding(
  model_root : StringView,
  claims : @instance_policy_file.TokenizerPolicyClaims,
  tokenizer_file : @json_file.TokenizerFileAdmission,
) -> (@tokenizer.TokenizerSpec, @tokenizer.TokenizerDigest) raise RuntimeInstanceAdmissionError {
  let digest = @tokenizer.TokenizerDigest::parse(claims.sha256()) catch {
    _ => raise reject(TokenizerBinding)
  }
  let file_limits = @json_file.TokenizerFileLimits::new(
    max_file_bytes=claims.max_file_bytes(),
    max_json_depth=claims.max_json_depth(),
  ) catch {
    _ => raise reject(TokenizerBinding)
  }
  let limits : @tokenizer.TokenizerLimits = {
    max_input_bytes: claims.max_input_bytes(),
    max_output_tokens: claims.max_output_tokens(),
    max_decoded_bytes: claims.max_decoded_bytes(),
    max_vocab_entries: claims.max_vocab_entries(),
    max_merge_rules: claims.max_merge_rules(),
    max_token_bytes: claims.max_token_bytes(),
    max_special_tokens: claims.max_special_tokens(),
  }
  if !tokenizer_file.matches_source(
      model_root,
      claims.locator(),
      digest,
      file_limits,
      limits,
    ) {
    raise reject(TokenizerBinding)
  }
  let spec = tokenizer_file.spec()
  if spec.digest() != digest || spec.limits() != limits {
    raise reject(TokenizerBinding)
  }
  (spec, digest)
}

///|
fn admit_scheduler(
  claims : @instance_policy_file.SchedulerPolicyClaims,
) -> @scheduler.SchedulerConfig raise RuntimeInstanceAdmissionError {
  @scheduler.new(
    step_token_budget=claims.step_token_budget(),
    max_active_requests=claims.max_active_requests(),
    max_waiting_requests=claims.max_waiting_requests(),
    prefill_chunk_tokens=claims.prefill_chunk_tokens(),
    emergency_decode_page_reserve=claims.emergency_decode_page_reserve(),
    waiting_age_threshold_steps=claims.waiting_age_threshold_steps(),
    output_event_capacity=claims.output_event_capacity(),
  ) catch {
    _ => raise reject(Scheduler)
  }
}

///|
fn admit_cache(
  claims : @instance_policy_file.CachePolicyClaims,
) -> @cache.CacheConfig raise RuntimeInstanceAdmissionError {
  @cache.new(
    tokens_per_page=claims.tokens_per_page(),
    total_page_count=claims.total_page_count(),
    block_table_pages_per_request=claims.block_table_pages_per_request(),
    prefix_enabled=claims.prefix_enabled(),
    max_prefix_entries=claims.max_prefix_entries(),
    max_prefix_nodes=claims.max_prefix_nodes(),
    max_prefix_tokens_per_entry=claims.max_prefix_tokens_per_entry(),
    max_prefix_pages=claims.max_prefix_pages(),
    max_active_references_per_page=claims.max_active_references_per_page(),
    max_cached_references_per_page=claims.max_cached_references_per_page(),
    layout_version=claims.layout_version(),
    max_prefix_scope_bytes=claims.max_prefix_scope_bytes(),
  ) catch {
    _ => raise reject(Cache)
  }
}

///|
fn admit_service(
  claims : @instance_policy_file.ServicePolicyClaims,
) -> @service.ServiceConfig raise RuntimeInstanceAdmissionError {
  let diagnostic_mode = match claims.diagnostic_mode() {
    "normal" => @service.Normal
    "verbose" => Verbose
    _ => raise reject(Service)
  }
  @service.new(
    claims.listen_host(),
    claims.listen_port(),
    claims.max_request_bytes(),
    claims.graceful_drain_milliseconds(),
    diagnostic_mode,
  ) catch {
    _ => raise reject(Service)
  }
}

///|
fn admit_worker_process(
  claims : @instance_policy_file.WorkerProcessPolicyClaims,
  worker : @worker_protocol.WorkerProtocolLimits,
) -> @worker_process.WorkerProcessLimits raise RuntimeInstanceAdmissionError {
  let limits = @worker_process.WorkerProcessLimits::new(
    max_frame_bytes=claims.max_frame_bytes(),
    startup_io_timeout_millis=claims.startup_io_timeout_millis(),
    io_timeout_millis=claims.io_timeout_millis(),
    shutdown_timeout_millis=claims.shutdown_timeout_millis(),
  ) catch {
    _ => raise reject(WorkerProcess)
  }
  let required = required_worker_frame_capacity(worker)
  if limits.max_frame_bytes() < required {
    raise reject(WorkerProcess)
  }
  limits
}

///|
fn admit_restart(
  claims : @instance_policy_file.RestartPolicyClaims,
) -> @worker_service.WorkerRestartBackoffPolicy raise RuntimeInstanceAdmissionError {
  @worker_service.WorkerRestartBackoffPolicy::new(
    initial_delay_millis=claims.initial_backoff_millis(),
    maximum_delay_millis=claims.maximum_backoff_millis(),
    stable_success_millis=claims.stable_after_millis(),
    maximum_attempts=claims.maximum_attempts(),
  ) catch {
    _ => raise reject(Restart)
  }
}

///|
fn admit_transport(
  claims : @instance_policy_file.TransportPolicyClaims,
) -> @online_tcp.LunaOnlineTcpLimits raise RuntimeInstanceAdmissionError {
  @online_tcp.LunaOnlineTcpLimits::new(
    read_chunk_bytes=claims.read_chunk_bytes(),
    write_chunk_bytes=claims.write_chunk_bytes(),
    accept_timeout_millis=claims.accept_timeout_millis(),
    input_idle_timeout_millis=claims.input_idle_timeout_millis(),
    write_timeout_millis=claims.write_timeout_millis(),
    reactor_transition_budget=claims.reactor_transition_budget(),
  ) catch {
    _ => raise reject(Transport)
  }
}
