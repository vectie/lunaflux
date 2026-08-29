///|
fn openai_responses_required_string(
  object : Map[String, Json],
  field : String,
) -> String raise OpenAIBaselineAdapterError {
  match object.get(field) {
    Some(String(value)) if !value.is_empty() => value
    _ => raise OpenAIBaselineAmbiguousEvent
  }
}

///|
fn openai_responses_require_id(
  lane : OpenAIResponsesLane,
  response_id : String,
) -> Unit raise OpenAIBaselineAdapterError {
  match lane.response_id {
    Some(expected) if expected == response_id => ()
    _ => raise OpenAIBaselineAmbiguousEvent
  }
}

///|
fn openai_responses_validate_usage(
  usage : Map[String, Json],
  declared_input_tokens : Int64,
) -> Int64 raise OpenAIBaselineAdapterError {
  let input = openai_baseline_usage_int(usage, "input_tokens")
  let cached = match usage.get("cached_input_tokens") {
    Some(value) => openai_baseline_exact_int(value)
    None => 0L
  }
  let output = openai_baseline_usage_int(usage, "output_tokens")
  let total = openai_baseline_usage_int(usage, "total_tokens")
  if input != declared_input_tokens ||
    cached < 0L ||
    cached > input ||
    output < 0L ||
    input > 9223372036854775807L - output ||
    total != input + output {
    raise OpenAIBaselineInvalidUsage
  }
  output
}

///|
fn OpenAIResponsesTrialObserver::observe_created(
  self : OpenAIResponsesTrialObserver,
  request_ordinal : Int,
  root : Map[String, Json],
) -> Unit raise OpenAIBaselineAdapterError {
  let lane = self.lanes[request_ordinal]
  if lane.created_seen {
    raise OpenAIBaselineAmbiguousEvent
  }
  guard root.get("response") is Some(Object(response)) else {
    raise OpenAIBaselineAmbiguousEvent
  }
  let response_id = openai_responses_required_string(response, "id")
  guard response.get("object") is Some(String("response")) &&
    response.get("status") is Some(String("in_progress")) else {
    raise OpenAIBaselineAmbiguousEvent
  }
  lane.response_id = Some(response_id)
  lane.created_seen = true
}

///|
fn OpenAIResponsesTrialObserver::observe_delta(
  self : OpenAIResponsesTrialObserver,
  request_ordinal : Int,
  root : Map[String, Json],
) -> Unit raise OpenAIBaselineAdapterError {
  let lane = self.lanes[request_ordinal]
  if !lane.created_seen || lane.usage_seen {
    raise OpenAIBaselineAmbiguousEvent
  }
  let response_id = openai_responses_required_string(root, "response_id")
  openai_responses_require_id(lane, response_id)
  guard root.get("output_index") is Some(output_index) &&
    root.get("content_index") is Some(content_index) &&
    root.get("delta") is Some(String(_)) else {
    raise OpenAIBaselineAmbiguousEvent
  }
  if openai_baseline_exact_int(output_index) != 0L ||
    openai_baseline_exact_int(content_index) != 0L {
    raise OpenAIBaselineAmbiguousEvent
  }
  if !lane.first_delta_seen {
    self.collector_first_token_observed(request_ordinal)
    lane.first_delta_seen = true
  }
  if lane.observed_delta_events == 9223372036854775807L {
    raise OpenAIBaselineInvalidUsage
  }
  lane.observed_delta_events += 1L
}

///|
fn OpenAIResponsesTrialObserver::observe_usage(
  self : OpenAIResponsesTrialObserver,
  request_ordinal : Int,
  root : Map[String, Json],
) -> Unit raise OpenAIBaselineAdapterError {
  let lane = self.lanes[request_ordinal]
  if !lane.created_seen || lane.usage_seen {
    raise OpenAIBaselineAmbiguousEvent
  }
  let response_id = openai_responses_required_string(root, "response_id")
  openai_responses_require_id(lane, response_id)
  guard root.get("usage") is Some(Object(usage)) else {
    raise OpenAIBaselineAmbiguousEvent
  }
  lane.completion_tokens = openai_responses_validate_usage(
    usage,
    self.inputs.request_input_tokens[request_ordinal],
  )
  lane.usage_seen = true
}

///|
fn OpenAIResponsesTrialObserver::observe_completed(
  self : OpenAIResponsesTrialObserver,
  request_ordinal : Int,
  root : Map[String, Json],
) -> Unit raise OpenAIBaselineAdapterError {
  let lane = self.lanes[request_ordinal]
  if !lane.created_seen {
    raise OpenAIBaselineAmbiguousEvent
  }
  guard root.get("response") is Some(Object(response)) else {
    raise OpenAIBaselineAmbiguousEvent
  }
  let response_id = openai_responses_required_string(response, "id")
  openai_responses_require_id(lane, response_id)
  guard response.get("object") is Some(String("response")) &&
    response.get("status") is Some(String("completed")) else {
    raise OpenAIBaselineAmbiguousEvent
  }
  guard response.get("usage") is Some(Object(usage)) else {
    raise OpenAIBaselineAmbiguousEvent
  }
  let output = openai_responses_validate_usage(
    usage,
    self.inputs.request_input_tokens[request_ordinal],
  )
  if lane.usage_seen && output != lane.completion_tokens {
    raise OpenAIBaselineInvalidUsage
  }
  lane.completion_tokens = output
  lane.usage_seen = true
  lane.terminal_kind = match response.get("finish_reason") {
    None => OPENAI_RESPONSES_COMPLETED
    Some(String("stop")) | Some(String("length")) => OPENAI_RESPONSES_COMPLETED
    Some(String("cancelled")) => OPENAI_RESPONSES_CANCELLED
    _ => raise OpenAIBaselineAmbiguousEvent
  }
}

///|
fn OpenAIResponsesTrialObserver::observe_failed(
  self : OpenAIResponsesTrialObserver,
  request_ordinal : Int,
  root : Map[String, Json],
) -> Unit raise OpenAIBaselineAdapterError {
  let lane = self.lanes[request_ordinal]
  if !lane.created_seen {
    raise OpenAIBaselineAmbiguousEvent
  }
  guard root.get("response") is Some(Object(response)) else {
    raise OpenAIBaselineAmbiguousEvent
  }
  guard response.get("status") is Some(String("failed")) else {
    raise OpenAIBaselineAmbiguousEvent
  }
  let response_id = openai_responses_required_string(response, "id")
  openai_responses_require_id(lane, response_id)
  match response.get("object") {
    None | Some(String("response")) => ()
    _ => raise OpenAIBaselineAmbiguousEvent
  }
  guard response.get("error") is Some(Object(error)) else {
    raise OpenAIBaselineAmbiguousEvent
  }
  if openai_responses_required_string(error, "code").is_empty() ||
    openai_responses_required_string(error, "message").is_empty() {
    raise OpenAIBaselineAmbiguousEvent
  }
  match response.get("usage") {
    Some(Object(usage)) => {
      let output = openai_responses_validate_usage(
        usage,
        self.inputs.request_input_tokens[request_ordinal],
      )
      if lane.usage_seen && output != lane.completion_tokens {
        raise OpenAIBaselineInvalidUsage
      }
      lane.completion_tokens = output
      lane.usage_seen = true
    }
    None | Some(Null) => ()
    _ => raise OpenAIBaselineAmbiguousEvent
  }
  lane.terminal_kind = OPENAI_RESPONSES_FAILED
}

///|
fn OpenAIResponsesTrialObserver::observe_auxiliary(
  self : OpenAIResponsesTrialObserver,
  request_ordinal : Int,
  root : Map[String, Json],
) -> Unit raise OpenAIBaselineAdapterError {
  let lane = self.lanes[request_ordinal]
  if !lane.created_seen || lane.usage_seen {
    raise OpenAIBaselineAmbiguousEvent
  }
  match root.get("response_id") {
    Some(String(response_id)) => openai_responses_require_id(lane, response_id)
    None =>
      match root.get("response") {
        Some(Object(response)) =>
          openai_responses_require_id(
            lane,
            openai_responses_required_string(response, "id"),
          )
        _ => raise OpenAIBaselineAmbiguousEvent
      }
    _ => raise OpenAIBaselineAmbiguousEvent
  }
  match root.get("output_index") {
    Some(value) if openai_baseline_exact_int(value) != 0L =>
      raise OpenAIBaselineAmbiguousEvent
    _ => ()
  }
  match root.get("content_index") {
    Some(value) if openai_baseline_exact_int(value) != 0L =>
      raise OpenAIBaselineAmbiguousEvent
    _ => ()
  }
}

///|
fn OpenAIResponsesTrialObserver::observe_json_event(
  self : OpenAIResponsesTrialObserver,
  request_ordinal : Int,
  document : StringView,
  declared_event : String?,
) -> Unit raise OpenAIBaselineAdapterError {
  @json_guard.reject_duplicate_object_keys(
    document,
    max_depth=self.maximum_json_depth,
  ) catch {
    _ => raise OpenAIBaselineInvalidSse
  }
  let parsed = @json.parse(document) catch {
    _ => raise OpenAIBaselineInvalidSse
  }
  guard parsed is Object(root) && root.get("type") is Some(String(kind)) else {
    raise OpenAIBaselineAmbiguousEvent
  }
  match declared_event {
    Some(declared) if declared != kind => raise OpenAIBaselineAmbiguousEvent
    _ => ()
  }
  match kind {
    "response.created" => self.observe_created(request_ordinal, root)
    "response.output_text.delta" => self.observe_delta(request_ordinal, root)
    "response.usage" => self.observe_usage(request_ordinal, root)
    "response.completed" => self.observe_completed(request_ordinal, root)
    "response.failed" => self.observe_failed(request_ordinal, root)
    "response.in_progress"
    | "response.output_item.added"
    | "response.content_part.added"
    | "response.output_text.done"
    | "response.content_part.done"
    | "response.output_item.done" =>
      self.observe_auxiliary(request_ordinal, root)
    _ => raise OpenAIBaselineAmbiguousEvent
  }
}
