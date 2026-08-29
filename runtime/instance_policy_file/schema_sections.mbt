///|
fn parse_service(
  root : Map[String, Json],
) -> ServicePolicyClaims raise InstancePolicyFileError {
  let object = require_object(root, "service", Root, index=5)
  validate_fields(object, Service, [
    "listen_host", "listen_port", "max_request_bytes", "graceful_drain_milliseconds",
    "diagnostic_mode",
  ])
  {
    listen_host: require_string(object, "listen_host", Service),
    listen_port: require_int(object, "listen_port", Service, index=1),
    max_request_bytes: require_int(
      object,
      "max_request_bytes",
      Service,
      index=2,
    ),
    graceful_drain_milliseconds: require_int(
      object,
      "graceful_drain_milliseconds",
      Service,
      index=3,
    ),
    diagnostic_mode: require_string(object, "diagnostic_mode", Service, index=4),
  }
}

///|
fn native_framed_protocol() -> ExternalProtocolPolicyClaims {
  try! ExternalProtocolPolicyClaims::new(
    NativeFramedV1,
    LoopbackPlaintext,
    DeploymentBearer,
    "",
    "",
    "",
    "",
    "",
  )
}

///|
fn parse_external_protocol(
  root : Map[String, Json],
) -> ExternalProtocolPolicyClaims raise InstancePolicyFileError {
  let object = require_object(root, "external_protocol", Root, index=6)
  validate_fields(object, ExternalProtocol, [
    "mode", "transport_security", "control_authentication", "invocation_path", "health_path",
    "readiness_path", "drain_path", "drain_method",
  ])
  let mode = match require_string(object, "mode", ExternalProtocol) {
    "native_framed_v1" => NativeFramedV1
    "openai_responses_v1" => OpenAiResponsesV1
    _ => raise invalid_field(ExternalProtocol, UnsupportedValue)
  }
  let transport_security = match
    require_string(object, "transport_security", ExternalProtocol, index=1) {
    "loopback_plaintext" => LoopbackPlaintext
    "private_network_plaintext" => PrivateNetworkPlaintext
    "deployment_tls" => DeploymentTls
    _ => raise invalid_field(ExternalProtocol, UnsupportedValue, index=1)
  }
  let control_authentication = match
    require_string(object, "control_authentication", ExternalProtocol, index=2) {
    "deployment_bearer" => DeploymentBearer
    _ => raise invalid_field(ExternalProtocol, UnsupportedValue, index=2)
  }
  let invocation_path = require_string(
    object,
    "invocation_path",
    ExternalProtocol,
    index=3,
  )
  let health_path = require_string(
    object,
    "health_path",
    ExternalProtocol,
    index=4,
  )
  let readiness_path = require_string(
    object,
    "readiness_path",
    ExternalProtocol,
    index=5,
  )
  let drain_path = require_string(
    object,
    "drain_path",
    ExternalProtocol,
    index=6,
  )
  let drain_method = require_string(
    object,
    "drain_method",
    ExternalProtocol,
    index=7,
  )
  ExternalProtocolPolicyClaims::new(
    mode, transport_security, control_authentication, invocation_path, health_path,
    readiness_path, drain_path, drain_method,
  ) catch {
    _ => raise invalid_field(ExternalProtocol, UnsupportedValue)
  }
}

///|
fn parse_openai_service(
  root : Map[String, Json],
) -> OpenAiServicePolicyClaims raise InstancePolicyFileError {
  let object = require_object(root, "openai_service", OpenAiService, index=7)
  validate_fields(object, OpenAiService, [
    "maximum_credential_bytes", "max_head_bytes", "max_body_bytes", "http_step_work_units",
    "response_step_work_units", "inbound_step_work_units", "outbound_step_work_units",
    "max_messages", "system_prefix", "system_suffix", "user_prefix", "user_suffix",
    "assistant_prefix", "assistant_suffix", "assistant_cue", "max_rendered_prompt_bytes",
    "model_alias", "response_id_prefix", "cache_scope_ascii", "max_new_tokens", "context_ceiling",
    "sampling_seed", "deadline_milliseconds",
  ])
  {
    maximum_credential_bytes: require_int(
      object,
      "maximum_credential_bytes",
      OpenAiService,
    ),
    max_head_bytes: require_int(
      object,
      "max_head_bytes",
      OpenAiService,
      index=1,
    ),
    max_body_bytes: require_int(
      object,
      "max_body_bytes",
      OpenAiService,
      index=2,
    ),
    http_step_work_units: require_int(
      object,
      "http_step_work_units",
      OpenAiService,
      index=3,
    ),
    response_step_work_units: require_int(
      object,
      "response_step_work_units",
      OpenAiService,
      index=4,
    ),
    inbound_step_work_units: require_int(
      object,
      "inbound_step_work_units",
      OpenAiService,
      index=5,
    ),
    outbound_step_work_units: require_int(
      object,
      "outbound_step_work_units",
      OpenAiService,
      index=6,
    ),
    max_messages: require_int(object, "max_messages", OpenAiService, index=7),
    system_prefix: require_string(
      object,
      "system_prefix",
      OpenAiService,
      index=8,
    ),
    system_suffix: require_string(
      object,
      "system_suffix",
      OpenAiService,
      index=9,
    ),
    user_prefix: require_string(object, "user_prefix", OpenAiService, index=10),
    user_suffix: require_string(object, "user_suffix", OpenAiService, index=11),
    assistant_prefix: require_string(
      object,
      "assistant_prefix",
      OpenAiService,
      index=12,
    ),
    assistant_suffix: require_string(
      object,
      "assistant_suffix",
      OpenAiService,
      index=13,
    ),
    assistant_cue: require_string(
      object,
      "assistant_cue",
      OpenAiService,
      index=14,
    ),
    max_rendered_prompt_bytes: require_int(
      object,
      "max_rendered_prompt_bytes",
      OpenAiService,
      index=15,
    ),
    model_alias: require_string(object, "model_alias", OpenAiService, index=16),
    response_id_prefix: require_string(
      object,
      "response_id_prefix",
      OpenAiService,
      index=17,
    ),
    cache_scope_ascii: require_string(
      object,
      "cache_scope_ascii",
      OpenAiService,
      index=18,
    ),
    max_new_tokens: require_int(
      object,
      "max_new_tokens",
      OpenAiService,
      index=19,
    ),
    context_ceiling: require_int(
      object,
      "context_ceiling",
      OpenAiService,
      index=20,
    ),
    sampling_seed: require_uint64(
      object,
      "sampling_seed",
      OpenAiService,
      index=21,
    ),
    deadline_milliseconds: require_uint64(
      object,
      "deadline_milliseconds",
      OpenAiService,
      index=22,
    ),
  }
}

///|
fn parse_worker_process(
  root : Map[String, Json],
) -> WorkerProcessPolicyClaims raise InstancePolicyFileError {
  let object = require_object(root, "worker_process", Root, index=6)
  validate_fields(object, WorkerProcess, [
    "max_frame_bytes", "startup_io_timeout_millis", "io_timeout_millis", "shutdown_timeout_millis",
  ])
  {
    max_frame_bytes: require_int(object, "max_frame_bytes", WorkerProcess),
    startup_io_timeout_millis: require_int(
      object,
      "startup_io_timeout_millis",
      WorkerProcess,
      index=1,
    ),
    io_timeout_millis: require_int(
      object,
      "io_timeout_millis",
      WorkerProcess,
      index=2,
    ),
    shutdown_timeout_millis: require_int(
      object,
      "shutdown_timeout_millis",
      WorkerProcess,
      index=3,
    ),
  }
}

///|
fn parse_restart(
  root : Map[String, Json],
) -> RestartPolicyClaims raise InstancePolicyFileError {
  let object = require_object(root, "restart", Root, index=7)
  validate_fields(object, Restart, [
    "initial_backoff_millis", "maximum_backoff_millis", "stable_after_millis", "maximum_attempts",
  ])
  {
    initial_backoff_millis: require_uint64(
      object,
      "initial_backoff_millis",
      Restart,
    ),
    maximum_backoff_millis: require_uint64(
      object,
      "maximum_backoff_millis",
      Restart,
      index=1,
    ),
    stable_after_millis: require_uint64(
      object,
      "stable_after_millis",
      Restart,
      index=2,
    ),
    maximum_attempts: require_int(object, "maximum_attempts", Restart, index=3),
  }
}

///|
fn parse_transport(
  root : Map[String, Json],
) -> TransportPolicyClaims raise InstancePolicyFileError {
  let object = require_object(root, "transport", Root, index=8)
  validate_fields(object, Transport, [
    "read_chunk_bytes", "write_chunk_bytes", "accept_timeout_millis", "input_idle_timeout_millis",
    "write_timeout_millis", "reactor_transition_budget",
  ])
  {
    read_chunk_bytes: require_int(object, "read_chunk_bytes", Transport),
    write_chunk_bytes: require_int(
      object,
      "write_chunk_bytes",
      Transport,
      index=1,
    ),
    accept_timeout_millis: require_int(
      object,
      "accept_timeout_millis",
      Transport,
      index=2,
    ),
    input_idle_timeout_millis: require_int(
      object,
      "input_idle_timeout_millis",
      Transport,
      index=3,
    ),
    write_timeout_millis: require_int(
      object,
      "write_timeout_millis",
      Transport,
      index=4,
    ),
    reactor_transition_budget: require_int(
      object,
      "reactor_transition_budget",
      Transport,
      index=5,
    ),
  }
}

///|
fn parse_preparation(
  root : Map[String, Json],
) -> PreparationPolicyClaims raise InstancePolicyFileError {
  let object = require_object(root, "preparation", Root, index=9)
  validate_fields(object, Preparation, [
    "framed_max_frame_bytes", "lane_count", "step_work_units", "total_work_units",
    "storage_int_cells", "storage_byte_cells", "storage_reference_cells", "event_work_units",
  ])
  {
    framed_max_frame_bytes: require_int(
      object,
      "framed_max_frame_bytes",
      Preparation,
    ),
    lane_count: require_int(object, "lane_count", Preparation, index=1),
    step_work_units: require_int(
      object,
      "step_work_units",
      Preparation,
      index=2,
    ),
    total_work_units: require_uint64(
      object,
      "total_work_units",
      Preparation,
      index=3,
    ),
    storage_int_cells: require_uint64(
      object,
      "storage_int_cells",
      Preparation,
      index=4,
    ),
    storage_byte_cells: require_uint64(
      object,
      "storage_byte_cells",
      Preparation,
      index=5,
    ),
    storage_reference_cells: require_uint64(
      object,
      "storage_reference_cells",
      Preparation,
      index=6,
    ),
    event_work_units: require_int(
      object,
      "event_work_units",
      Preparation,
      index=7,
    ),
  }
}

///|
fn parse_telemetry(
  root : Map[String, Json],
) -> TelemetryPolicyClaims raise InstancePolicyFileError {
  let object = require_object(root, "telemetry", Root, index=10)
  validate_fields(object, Telemetry, ["instance_log_capacity"])
  {
    instance_log_capacity: require_int(
      object,
      "instance_log_capacity",
      Telemetry,
    ),
  }
}
