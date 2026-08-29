///|
/// Independent lowercase SHA-256 identity of the exact policy bytes.
pub struct InstancePolicyDigest(String) derive(Debug, Eq)

///|
/// Startup ceilings for one policy snapshot and structural scan.
pub struct InstancePolicyFileLimits {
  priv max_policy_bytes : Int64
  priv max_json_depth : Int
}

///|
/// Strict tokenizer artifact binding and bounded parser/use limits.
pub struct TokenizerPolicyClaims {
  priv locator : String
  priv sha256 : String
  priv max_file_bytes : Int
  priv max_json_depth : Int
  priv max_input_bytes : Int
  priv max_output_tokens : Int
  priv max_decoded_bytes : Int
  priv max_vocab_entries : Int
  priv max_merge_rules : Int
  priv max_token_bytes : Int
  priv max_special_tokens : Int
} derive(Debug, Eq)

///|
/// Exact SchedulerConfig constructor claims.
pub struct SchedulerPolicyClaims {
  priv step_token_budget : Int
  priv max_active_requests : Int
  priv max_waiting_requests : Int
  priv prefill_chunk_tokens : Int
  priv emergency_decode_page_reserve : Int
  priv waiting_age_threshold_steps : Int
  priv output_event_capacity : Int
} derive(Debug, Eq)

///|
/// Exact CacheConfig constructor claims.
pub struct CachePolicyClaims {
  priv tokens_per_page : Int
  priv total_page_count : Int
  priv block_table_pages_per_request : Int
  priv prefix_enabled : Bool
  priv max_prefix_entries : Int
  priv max_prefix_nodes : Int
  priv max_prefix_tokens_per_entry : Int
  priv max_prefix_pages : Int
  priv max_prefix_scope_bytes : Int
  priv max_active_references_per_page : Int
  priv max_cached_references_per_page : Int
  priv layout_version : Int
} derive(Debug, Eq)

///|
/// Exact native ServiceConfig constructor claims.
pub struct ServicePolicyClaims {
  priv listen_host : String
  priv listen_port : Int
  priv max_request_bytes : Int
  priv graceful_drain_milliseconds : Int
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
  priv maximum_credential_bytes : Int
  priv max_head_bytes : Int
  priv max_body_bytes : Int
  priv http_step_work_units : Int
  priv response_step_work_units : Int
  priv inbound_step_work_units : Int
  priv outbound_step_work_units : Int
  priv max_messages : Int
  priv system_prefix : String
  priv system_suffix : String
  priv user_prefix : String
  priv user_suffix : String
  priv assistant_prefix : String
  priv assistant_suffix : String
  priv assistant_cue : String
  priv max_rendered_prompt_bytes : Int
  priv model_alias : String
  priv response_id_prefix : String
  priv cache_scope_ascii : String
  priv max_new_tokens : Int
  priv context_ceiling : Int
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
  priv max_frame_bytes : Int
  priv startup_io_timeout_millis : Int
  priv io_timeout_millis : Int
  priv shutdown_timeout_millis : Int
} derive(Debug, Eq)

///|
/// Exact worker-restart policy claims.
pub struct RestartPolicyClaims {
  priv initial_backoff_millis : UInt64
  priv maximum_backoff_millis : UInt64
  priv stable_after_millis : UInt64
  priv maximum_attempts : Int
} derive(Debug, Eq)

///|
/// Exact native TCP transport claims.
pub struct TransportPolicyClaims {
  priv read_chunk_bytes : Int
  priv write_chunk_bytes : Int
  priv accept_timeout_millis : Int
  priv input_idle_timeout_millis : Int
  priv write_timeout_millis : Int
  priv reactor_transition_budget : Int
} derive(Debug, Eq)

///|
/// Exact bounded request/event preparation claims.
pub struct PreparationPolicyClaims {
  priv framed_max_frame_bytes : Int
  priv lane_count : Int
  priv step_work_units : Int
  priv total_work_units : UInt64
  priv storage_int_cells : UInt64
  priv storage_byte_cells : UInt64
  priv storage_reference_cells : UInt64
  priv event_work_units : Int
} derive(Debug, Eq)

///|
/// Exact bounded instance-telemetry claims.
pub struct TelemetryPolicyClaims {
  priv instance_log_capacity : Int
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
