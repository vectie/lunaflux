///|
fn parse_policy(
  bytes : BytesView,
  digest : InstancePolicyDigest,
  limits : InstancePolicyFileLimits,
) -> InstancePolicyAdmission raise InstancePolicyFileError {
  if bytes.length().to_int64() > limits.max_policy_bytes() {
    raise FileTooLarge(maximum=limits.max_policy_bytes())
  }
  let text = @utf8.decode(bytes) catch { _ => raise InvalidUtf8 }
  @json_guard.reject_duplicate_object_keys(
    text,
    max_depth=limits.max_json_depth(),
  ) catch {
    InvalidDepthLimit => raise MalformedJson
    DepthLimitExceeded(_) =>
      raise JsonDepthExceeded(maximum=limits.max_json_depth())
    DuplicateObjectKey(depth~, index~) => raise DuplicateField(depth~, index~)
    MalformedStructure => raise MalformedJson
  }
  let value = @json.parse(text) catch { _ => raise MalformedJson }
  guard value is Object(root) else { raise MalformedJson }
  let schema_version = require_string(root, "schema_version", Root)
  let (external_protocol, openai_service) = match schema_version {
    "lunaflux.instance-policy.v1" => {
      validate_fields(root, Root, [
        "schema_version", "runtime_descriptor_sha256", "tokenizer", "scheduler",
        "cache", "service", "worker_process", "restart", "transport", "preparation",
        "telemetry",
      ])
      (native_framed_protocol(), None)
    }
    "lunaflux.instance-policy.v2" => {
      validate_fields(root, Root, [
        "schema_version", "runtime_descriptor_sha256", "tokenizer", "scheduler",
        "cache", "service", "external_protocol", "worker_process", "restart", "transport",
        "preparation", "telemetry",
      ])
      (parse_external_protocol(root), None)
    }
    "lunaflux.instance-policy.v3" => {
      validate_fields(root, Root, [
        "schema_version", "runtime_descriptor_sha256", "tokenizer", "scheduler",
        "cache", "service", "external_protocol", "openai_service", "worker_process",
        "restart", "transport", "preparation", "telemetry",
      ])
      let protocol = parse_external_protocol(root)
      if protocol.mode() != OpenAiResponsesV1 ||
        (
          protocol.transport_security() != LoopbackPlaintext &&
          protocol.transport_security() != PrivateNetworkPlaintext
        ) {
        raise invalid_field(ExternalProtocol, UnsupportedValue)
      }
      (protocol, Some(parse_openai_service(root)))
    }
    _ => raise invalid_field(Root, UnsupportedValue)
  }
  {
    digest,
    runtime_descriptor_sha256: require_lower_sha256(
      root,
      "runtime_descriptor_sha256",
      Root,
      1,
    ),
    tokenizer: parse_tokenizer(root),
    scheduler: parse_scheduler(root),
    cache: parse_cache(root),
    service: parse_service(root),
    external_protocol,
    openai_service,
    worker_process: parse_worker_process(root),
    restart: parse_restart(root),
    transport: parse_transport(root),
    preparation: parse_preparation(root),
    telemetry: parse_telemetry(root),
  }
}

///|
fn parse_tokenizer(
  root : Map[String, Json],
) -> TokenizerPolicyClaims raise InstancePolicyFileError {
  let object = require_object(root, "tokenizer", Root, index=2)
  validate_fields(object, Tokenizer, [
    "locator", "sha256", "max_file_bytes", "max_json_depth", "max_input_bytes", "max_output_tokens",
    "max_decoded_bytes", "max_vocab_entries", "max_merge_rules", "max_token_bytes",
    "max_special_tokens",
  ])
  {
    locator: require_relative_locator(object, "locator", Tokenizer, 0),
    sha256: require_lower_sha256(object, "sha256", Tokenizer, 1),
    max_file_bytes: require_int(object, "max_file_bytes", Tokenizer, index=2),
    max_json_depth: require_int(object, "max_json_depth", Tokenizer, index=3),
    max_input_bytes: require_int(object, "max_input_bytes", Tokenizer, index=4),
    max_output_tokens: require_int(
      object,
      "max_output_tokens",
      Tokenizer,
      index=5,
    ),
    max_decoded_bytes: require_int(
      object,
      "max_decoded_bytes",
      Tokenizer,
      index=6,
    ),
    max_vocab_entries: require_int(
      object,
      "max_vocab_entries",
      Tokenizer,
      index=7,
    ),
    max_merge_rules: require_int(object, "max_merge_rules", Tokenizer, index=8),
    max_token_bytes: require_int(object, "max_token_bytes", Tokenizer, index=9),
    max_special_tokens: require_int(
      object,
      "max_special_tokens",
      Tokenizer,
      index=10,
    ),
  }
}

///|
fn parse_scheduler(
  root : Map[String, Json],
) -> SchedulerPolicyClaims raise InstancePolicyFileError {
  let object = require_object(root, "scheduler", Root, index=3)
  validate_fields(object, Scheduler, [
    "step_token_budget", "max_active_requests", "max_waiting_requests", "prefill_chunk_tokens",
    "emergency_decode_page_reserve", "waiting_age_threshold_steps", "output_event_capacity",
  ])
  {
    step_token_budget: require_int(object, "step_token_budget", Scheduler),
    max_active_requests: require_int(
      object,
      "max_active_requests",
      Scheduler,
      index=1,
    ),
    max_waiting_requests: require_int(
      object,
      "max_waiting_requests",
      Scheduler,
      index=2,
    ),
    prefill_chunk_tokens: require_int(
      object,
      "prefill_chunk_tokens",
      Scheduler,
      index=3,
    ),
    emergency_decode_page_reserve: require_int(
      object,
      "emergency_decode_page_reserve",
      Scheduler,
      index=4,
    ),
    waiting_age_threshold_steps: require_int(
      object,
      "waiting_age_threshold_steps",
      Scheduler,
      index=5,
    ),
    output_event_capacity: require_int(
      object,
      "output_event_capacity",
      Scheduler,
      index=6,
    ),
  }
}

///|
fn parse_cache(
  root : Map[String, Json],
) -> CachePolicyClaims raise InstancePolicyFileError {
  let object = require_object(root, "cache", Root, index=4)
  validate_fields(object, Cache, [
    "tokens_per_page", "total_page_count", "block_table_pages_per_request", "prefix_enabled",
    "max_prefix_entries", "max_prefix_nodes", "max_prefix_tokens_per_entry", "max_prefix_pages",
    "max_prefix_scope_bytes", "max_active_references_per_page", "max_cached_references_per_page",
    "layout_version",
  ])
  {
    tokens_per_page: require_int(object, "tokens_per_page", Cache),
    total_page_count: require_int(object, "total_page_count", Cache, index=1),
    block_table_pages_per_request: require_int(
      object,
      "block_table_pages_per_request",
      Cache,
      index=2,
    ),
    prefix_enabled: require_bool(object, "prefix_enabled", Cache, index=3),
    max_prefix_entries: require_int(
      object,
      "max_prefix_entries",
      Cache,
      index=4,
    ),
    max_prefix_nodes: require_int(object, "max_prefix_nodes", Cache, index=5),
    max_prefix_tokens_per_entry: require_int(
      object,
      "max_prefix_tokens_per_entry",
      Cache,
      index=6,
    ),
    max_prefix_pages: require_int(object, "max_prefix_pages", Cache, index=7),
    max_prefix_scope_bytes: require_int(
      object,
      "max_prefix_scope_bytes",
      Cache,
      index=8,
    ),
    max_active_references_per_page: require_int(
      object,
      "max_active_references_per_page",
      Cache,
      index=9,
    ),
    max_cached_references_per_page: require_int(
      object,
      "max_cached_references_per_page",
      Cache,
      index=10,
    ),
    layout_version: require_int(object, "layout_version", Cache, index=11),
  }
}
