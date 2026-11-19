///|
/// Supplied lowercase SHA-256-format policy label, not a verified checksum.
pub struct InstancePolicyDigest(String) derive(Debug, Eq)

///|
/// Startup ceilings for one policy snapshot and structural scan.
pub struct InstancePolicyFileLimits {
  priv max_policy_bytes : Int64
  priv max_json_depth : Int32
}

///|
/// Strict tokenizer artifact binding and bounded parser/use limits.
pub struct TokenizerPolicyClaims {
  priv locator : String
  priv sha256 : String
  priv max_file_bytes : Int32
  priv max_json_depth : Int32
  priv max_input_bytes : Int32
  priv max_output_tokens : Int32
  priv max_decoded_bytes : Int32
  priv max_vocab_entries : Int32
  priv max_merge_rules : Int32
  priv max_token_bytes : Int32
  priv max_special_tokens : Int32
} derive(Debug, Eq)

///|
/// Exact SchedulerConfig constructor claims.
pub struct SchedulerPolicyClaims {
  priv step_token_budget : Int32
  priv max_active_requests : Int32
  priv max_waiting_requests : Int32
  priv prefill_chunk_tokens : Int32
  priv emergency_decode_page_reserve : Int32
  priv waiting_age_threshold_steps : Int32
  priv output_event_capacity : Int32
} derive(Debug, Eq)

///|
/// Exact CacheConfig constructor claims.
pub struct CachePolicyClaims {
  priv tokens_per_page : Int32
  priv total_page_count : Int32
  priv block_table_pages_per_request : Int32
  priv prefix_enabled : Bool
  priv max_prefix_entries : Int32
  priv max_prefix_nodes : Int32
  priv max_prefix_tokens_per_entry : Int32
  priv max_prefix_pages : Int32
  priv max_prefix_scope_bytes : Int32
  priv max_active_references_per_page : Int32
  priv max_cached_references_per_page : Int32
  priv layout_version : Int32
} derive(Debug, Eq)

///|
/// Exact native ServiceConfig constructor claims.
pub struct ServicePolicyClaims {
  priv listen_host : String
  priv listen_port : Int32
  priv max_request_bytes : Int32
  priv graceful_drain_milliseconds : Int32
  priv diagnostic_mode : String
} derive(Debug, Eq)

///|
/// Exact external protocol selected by the authenticated instance policy.
pub(all) enum ExternalProtocolMode {
  NativeFramedV1
  OpenAiResponsesV1
} derive(Debug, Eq)

///|
/// Transport ownership remains explicit. Loopback plaintext is not authority
/// for public exposure; deployment TLS requires a separately approved ingress.
pub(all) enum ExternalTransportSecurity {
  LoopbackPlaintext
  PrivateNetworkPlaintext
  DeploymentTls
} derive(Debug, Eq)

///|
/// LunaFlux retains no deployment credential in the instance policy.
pub(all) enum ExternalControlAuthentication {
  DeploymentBearer
} derive(Debug, Eq)

///|
/// Digest-bound endpoint intent. These claims own no socket, credential, TLS
/// material, service owner, readiness state, or routing authority.
pub struct ExternalProtocolPolicyClaims {
  priv mode : ExternalProtocolMode
  priv transport_security : ExternalTransportSecurity
  priv control_authentication : ExternalControlAuthentication
  priv invocation_path : String
  priv health_path : String
  priv readiness_path : String
  priv drain_path : String
  priv drain_method : String
}

///|
/// Digest-bound OpenAI Responses construction inputs. This owns no secret,
/// socket, TLS material, service owner, or routing authority.
pub struct OpenAiServicePolicyClaims {
  priv maximum_credential_bytes : Int32
  priv max_head_bytes : Int32
  priv max_body_bytes : Int32
  priv http_step_work_units : Int32
  priv response_step_work_units : Int32
  priv inbound_step_work_units : Int32
  priv outbound_step_work_units : Int32
  priv max_messages : Int32
  priv system_prefix : String
  priv system_suffix : String
  priv user_prefix : String
  priv user_suffix : String
  priv assistant_prefix : String
  priv assistant_suffix : String
  priv assistant_cue : String
  priv max_rendered_prompt_bytes : Int32
  priv model_alias : String
  priv response_id_prefix : String
  priv cache_scope_ascii : String
  priv max_new_tokens : Int32
  priv context_ceiling : Int32
  priv sampling_seed : UInt64
  priv deadline_milliseconds : UInt64
}

///|
pub(all) suberror ExternalProtocolPolicyError {
  InvalidExternalProtocolPolicy
} derive(Debug, Eq)

///|
/// Exact worker-process envelope claims.
pub struct WorkerProcessPolicyClaims {
  priv max_frame_bytes : Int32
  priv startup_io_timeout_millis : Int32
  priv io_timeout_millis : Int32
  priv shutdown_timeout_millis : Int32
} derive(Debug, Eq)

///|
/// Exact worker-restart policy claims.
pub struct RestartPolicyClaims {
  priv initial_backoff_millis : UInt64
  priv maximum_backoff_millis : UInt64
  priv stable_after_millis : UInt64
  priv maximum_attempts : Int32
} derive(Debug, Eq)

///|
/// Exact native TCP transport claims.
pub struct TransportPolicyClaims {
  priv read_chunk_bytes : Int32
  priv write_chunk_bytes : Int32
  priv accept_timeout_millis : Int32
  priv input_idle_timeout_millis : Int32
  priv write_timeout_millis : Int32
  priv reactor_transition_budget : Int32
} derive(Debug, Eq)

///|
/// Exact bounded request/event preparation claims.
pub struct PreparationPolicyClaims {
  priv framed_max_frame_bytes : Int32
  priv lane_count : Int32
  priv step_work_units : Int32
  priv total_work_units : UInt64
  priv storage_int_cells : UInt64
  priv storage_byte_cells : UInt64
  priv storage_reference_cells : UInt64
  priv event_work_units : Int32
} derive(Debug, Eq)

///|
/// Exact bounded instance-telemetry claims.
pub struct TelemetryPolicyClaims {
  priv instance_log_capacity : Int32
} derive(Debug, Eq)

///|
/// Root-free syntactic admission of one exact instance-policy document.
pub struct InstancePolicyAdmission {
  priv digest : InstancePolicyDigest
  priv runtime_descriptor_sha256 : String
  priv tokenizer : TokenizerPolicyClaims
  priv scheduler : SchedulerPolicyClaims
  priv cache : CachePolicyClaims
  priv service : ServicePolicyClaims
  priv external_protocol : ExternalProtocolPolicyClaims
  priv openai_service : OpenAiServicePolicyClaims?
  priv worker_process : WorkerProcessPolicyClaims
  priv restart : RestartPolicyClaims
  priv transport : TransportPolicyClaims
  priv preparation : PreparationPolicyClaims
  priv telemetry : TelemetryPolicyClaims
}
