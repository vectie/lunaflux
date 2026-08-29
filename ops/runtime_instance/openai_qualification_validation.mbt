///|
priv struct OpenAiQualificationMetricSample {
  admissions : UInt64
  completions : UInt64
  cancellations : UInt64
  deadlines : UInt64
  failures : UInt64
  prompt_tokens : UInt64
  generated_tokens : UInt64
  worker_failures : UInt64
  network_accepts : UInt64
  network_disconnects : UInt64
  network_rejections : UInt64
  queue_depth : Int
  active_requests : Int
  kv_pages_used : Int
  kv_pages_free : Int
}

///|
fn capture_openai_qualification_metrics(
  owner : RuntimeInstanceOwner,
) -> OpenAiQualificationMetricSample raise RuntimeOpenAiQualificationValidationError {
  let snapshot = owner.metrics_snapshot() catch {
    _ => raise openai_qualification_validation_error(OpenAiQualificationMetrics)
  }
  {
    admissions: snapshot.counter_value(Admissions),
    completions: snapshot.counter_value(Completions),
    cancellations: snapshot.counter_value(Cancellations),
    deadlines: snapshot.counter_value(Deadlines),
    failures: snapshot.counter_value(Failures),
    prompt_tokens: snapshot.counter_value(PromptTokens),
    generated_tokens: snapshot.counter_value(GeneratedTokens),
    worker_failures: snapshot.counter_value(WorkerFailures),
    network_accepts: snapshot.counter_value(NetworkAccepts),
    network_disconnects: snapshot.counter_value(NetworkDisconnects),
    network_rejections: snapshot.counter_value(NetworkRejections),
    queue_depth: snapshot.gauge_value(QueueDepth),
    active_requests: snapshot.gauge_value(ActiveRequests),
    kv_pages_used: snapshot.gauge_value(KvPagesUsed),
    kv_pages_free: snapshot.gauge_value(KvPagesFree),
  } catch {
    _ => raise openai_qualification_validation_error(OpenAiQualificationMetrics)
  }
}

///|
fn openai_qualification_initial_metrics_valid(
  sample : OpenAiQualificationMetricSample,
) -> Bool {
  sample.admissions == 0UL &&
  sample.completions == 0UL &&
  sample.cancellations == 0UL &&
  sample.deadlines == 0UL &&
  sample.failures == 0UL &&
  sample.prompt_tokens == 0UL &&
  sample.generated_tokens == 0UL &&
  sample.worker_failures == 0UL &&
  sample.network_accepts == 0UL &&
  sample.network_disconnects == 0UL &&
  sample.network_rejections == 0UL &&
  sample.queue_depth == 0 &&
  sample.active_requests == 0 &&
  sample.kv_pages_used == 0 &&
  sample.kv_pages_free > 0
}

///|
fn openai_qualification_terminal_metrics_valid(
  sample : OpenAiQualificationMetricSample,
  initial_kv_pages_free : Int,
) -> Bool {
  sample.admissions == 1UL &&
  sample.completions == 1UL &&
  sample.cancellations == 0UL &&
  sample.deadlines == 0UL &&
  sample.failures == 0UL &&
  sample.prompt_tokens > 0UL &&
  sample.generated_tokens == 2UL &&
  sample.worker_failures == 0UL &&
  sample.network_accepts == 3UL &&
  sample.network_disconnects == 3UL &&
  sample.network_rejections == 2UL &&
  sample.queue_depth == 0 &&
  sample.active_requests == 0 &&
  sample.kv_pages_used == 0 &&
  sample.kv_pages_free == initial_kv_pages_free
}

///|
fn openai_qualification_error_response(response : String) -> Bool {
  response.contains("HTTP/1.1 401 Unauthorized") &&
  response.contains("Content-Type: application/json") &&
  !response.contains("text/event-stream") &&
  !response.contains("data:")
}

///|
async fn refuse_openai_post_drain_connection(
  owner : RuntimeInstanceOwner,
  timeout_millis : Int,
) -> Bool {
  owner.begin_drain() catch {
    _ => return false
  }
  if owner.phase() != Draining || owner.readiness() != RuntimeNotReady {
    return false
  }
  ignore(owner.progress() catch { _ => return false })
  let connected = try
    @async.with_timeout_opt(timeout_millis, () => {
      @socket.Tcp::connect(owner.openai_servers[0].local_addr())
    })
  catch {
    _ => return true
  } noraise {
    value => value
  }
  match connected {
    None => true
    Some(client) => {
      client.close()
      false
    }
  }
}

///|
async fn drain_openai_qualification_owner(
  owner : RuntimeInstanceOwner,
  turn_limit : Int,
) -> Bool {
  if owner.cleanup_complete() {
    return true
  }
  match owner.phase() {
    Starting | HealthyNotReady | Ready | OpenAiQualificationReady =>
      owner.begin_drain() catch {
        _ => ()
      }
    Draining | Failed | Closed => ()
  }
  for turn in 0..<turn_limit {
    if owner.cleanup_complete() {
      return true
    }
    ignore(owner.progress() catch { _ => RuntimeFailedProgress })
    if turn % 64 == 63 {
      @async.pause()
    }
  }
  owner.cleanup_complete()
}

///|
async fn run_openai_qualification_validation(
  owner : RuntimeInstanceOwner,
  policy : RuntimeOpenAiQualificationPolicy,
  contract : RuntimeOpenAiQualificationValidationContract,
) -> RuntimeOpenAiQualificationValidationResult raise RuntimeOpenAiQualificationValidationError {
  if policy.server.max_new_tokens != 2 || policy.server.context_ceiling < 4 {
    raise openai_qualification_validation_error(OpenAiQualificationContract)
  }
  owner.bind_openai_qualification(policy) catch {
    _ => raise openai_qualification_validation_error(OpenAiQualificationOwner)
  }
  if owner.phase() != OpenAiQualificationReady ||
    owner.health() != RuntimeHealthy ||
    owner.readiness() != RuntimeNotReady ||
    owner.openai_qualification_readiness() != OpenAiQualificationListenerReady {
    raise openai_qualification_validation_error(OpenAiQualificationOwner)
  }
  let health = qualification_operational_exchange(owner, "/healthz", contract)
  if !health.contains("HTTP/1.1 200 OK") ||
    !health.has_suffix("{\"status\":\"healthy\"}\n") {
    raise openai_qualification_validation_error(OpenAiQualificationResponse)
  }
  let readiness = qualification_operational_exchange(owner, "/readyz", contract)
  if !readiness.contains("HTTP/1.1 503 Service Unavailable") ||
    !readiness.has_suffix("{\"status\":\"not_ready\"}\n") {
    raise openai_qualification_validation_error(OpenAiQualificationResponse)
  }
  let initial = capture_openai_qualification_metrics(owner)
  if !openai_qualification_initial_metrics_valid(initial) {
    raise openai_qualification_validation_error(OpenAiQualificationMetrics)
  }
  let model = qualification_decode(policy.server.model_alias)
  let credential = qualification_decode(contract.credential)
  let wrong = qualification_decode(
    qualification_wrong_credential(contract.credential),
  )
  let body = "{\"model\":\"" +
    model +
    "\",\"input\":\"\\u0001\\u0002\",\"stream\":true}"
  let missing_response = qualification_exchange(
    owner,
    qualification_http_request(None, body),
    1UL,
    contract,
  )
  if !openai_qualification_error_response(missing_response) {
    raise openai_qualification_validation_error(
      OpenAiQualificationAuthentication,
    )
  }
  let wrong_response = qualification_exchange(
    owner,
    qualification_http_request(Some(wrong), body),
    2UL,
    contract,
  )
  if !openai_qualification_error_response(wrong_response) {
    raise openai_qualification_validation_error(
      OpenAiQualificationAuthentication,
    )
  }
  let response = qualification_exchange(
    owner,
    qualification_http_request(Some(credential), body),
    3UL,
    contract,
  )
  if !response.contains("HTTP/1.1 200 OK") ||
    !response.contains("Content-Type: text/event-stream") {
    raise openai_qualification_validation_error(OpenAiQualificationResponse)
  }
  if !qualification_markers_ordered(response) {
    raise openai_qualification_validation_error(OpenAiQualificationEventOrder)
  }
  let terminal = capture_openai_qualification_metrics(owner)
  if !openai_qualification_terminal_metrics_valid(
      terminal,
      initial.kv_pages_free,
    ) {
    raise openai_qualification_validation_error(OpenAiQualificationMetrics)
  }
  if !(refuse_openai_post_drain_connection(owner, contract.io_timeout_millis) catch {
      _ => false
    }) {
    raise openai_qualification_validation_error(
      OpenAiQualificationPostDrainAdmission,
    )
  }
  {
    event_count: 5,
    network_accepts: terminal.network_accepts,
    network_disconnects: terminal.network_disconnects,
    network_rejections: terminal.network_rejections,
    admissions: terminal.admissions,
    completions: terminal.completions,
    kv_pages_free: terminal.kv_pages_free,
  }
}

///|
/// Runs one bounded caller-authorized loopback Responses campaign and returns
/// evidence only after deterministic child/listener/service cleanup.
pub async fn RuntimeInstanceOwner::validate_openai_qualification(
  self : RuntimeInstanceOwner,
  policy : RuntimeOpenAiQualificationPolicy,
  contract : RuntimeOpenAiQualificationValidationContract,
) -> RuntimeOpenAiQualificationValidationResult {
  @async.protect_from_cancel(() => {
    let outcome = try
      run_openai_qualification_validation(self, policy, contract)
    catch {
      RuntimeOpenAiQualificationValidationFailed(rule) => Err(rule)
    } noraise {
      result => Ok(result)
    }
    let cleaned = drain_openai_qualification_owner(self, contract.turn_limit)
    if !cleaned ||
      self.phase() != Closed ||
      self.health() != RuntimeClosed ||
      self.readiness() != RuntimeNotReady ||
      !self.cleanup_complete() {
      raise openai_qualification_validation_error(OpenAiQualificationCleanup)
    }
    match outcome {
      Ok(result) => result
      Err(rule) => raise openai_qualification_validation_error(rule)
    }
  })
}
