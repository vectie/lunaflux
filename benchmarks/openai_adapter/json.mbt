///|
fn openai_baseline_exact_int(
  value : Json,
) -> Int64 raise OpenAIBaselineAdapterError {
  guard value is Number(number, repr~) else {
    raise OpenAIBaselineAmbiguousEvent
  }
  match repr {
    Some(encoded) => {
      let parsed : Int64 = @strconv.from_str(encoded) catch {
        _ => raise OpenAIBaselineAmbiguousEvent
      }
      parsed
    }
    None => {
      if number.is_nan() || number.is_inf() {
        raise OpenAIBaselineAmbiguousEvent
      }
      let parsed = number.to_int64()
      if parsed.to_double() != number {
        raise OpenAIBaselineAmbiguousEvent
      }
      parsed
    }
  }
}

///|
fn openai_baseline_optional_usage(
  root : Map[String, Json],
) -> Map[String, Json]? raise OpenAIBaselineAdapterError {
  match root.get("usage") {
    None | Some(Null) => None
    Some(Object(usage)) => Some(usage)
    Some(_) => raise OpenAIBaselineAmbiguousEvent
  }
}

///|
fn openai_baseline_usage_int(
  usage : Map[String, Json],
  field : String,
) -> Int64 raise OpenAIBaselineAdapterError {
  match usage.get(field) {
    Some(value) => openai_baseline_exact_int(value)
    None => raise OpenAIBaselineInvalidUsage
  }
}

///|
fn OpenAIBaselineTrialAdapter::observe_usage(
  self : OpenAIBaselineTrialAdapter,
  request_ordinal : Int,
  usage : Map[String, Json],
) -> Unit raise OpenAIBaselineAdapterError {
  let lane = self.lanes[request_ordinal]
  if lane.usage_seen || !lane.finish_seen {
    raise OpenAIBaselineAmbiguousEvent
  }
  let prompt = openai_baseline_usage_int(usage, "prompt_tokens")
  let completion = openai_baseline_usage_int(usage, "completion_tokens")
  let total = openai_baseline_usage_int(usage, "total_tokens")
  let declared = self.inputs.request_input_tokens[request_ordinal]
  if prompt != declared ||
    completion < 0L ||
    prompt > 9223372036854775807L - completion ||
    total != prompt + completion ||
    completion != lane.observed_tokens ||
    (lane.first_content_seen && completion <= 0L) ||
    (!lane.first_content_seen && completion != 0L) {
    raise OpenAIBaselineInvalidUsage
  }
  lane.completion_tokens = completion
  lane.usage_seen = true
}

///|
fn openai_baseline_delta_content(
  delta : Map[String, Json],
) -> String? raise OpenAIBaselineAdapterError {
  for name, _ in delta {
    if name != "role" && name != "content" {
      raise OpenAIBaselineAmbiguousEvent
    }
  }
  match delta.get("role") {
    None => ()
    Some(String("assistant")) => ()
    _ => raise OpenAIBaselineAmbiguousEvent
  }
  match delta.get("content") {
    None | Some(Null) => None
    Some(String(content)) => Some(content)
    Some(_) => raise OpenAIBaselineAmbiguousEvent
  }
}

///|
fn openai_baseline_choice_token_count(
  choice : Map[String, Json],
  has_content : Bool,
) -> Int64 raise OpenAIBaselineAdapterError {
  match choice.get("logprobs") {
    None | Some(Null) =>
      if has_content {
        raise OpenAIBaselineAmbiguousEvent
      } else {
        0L
      }
    Some(Object(logprobs)) => {
      for name, _ in logprobs {
        if name != "content" {
          raise OpenAIBaselineAmbiguousEvent
        }
      }
      let entries = match logprobs.get("content") {
        Some(Array(values)) => values
        _ => raise OpenAIBaselineAmbiguousEvent
      }
      if entries.is_empty() != !has_content {
        raise OpenAIBaselineAmbiguousEvent
      }
      for entry in entries {
        guard entry is Object(token) && token.get("token") is Some(String(_)) else {
          raise OpenAIBaselineAmbiguousEvent
        }
      }
      Int64::from_int(entries.length())
    }
    Some(_) => raise OpenAIBaselineAmbiguousEvent
  }
}

///|
fn OpenAIBaselineTrialAdapter::observe_choice(
  self : OpenAIBaselineTrialAdapter,
  request_ordinal : Int,
  choice : Map[String, Json],
) -> Unit raise OpenAIBaselineAdapterError {
  let lane = self.lanes[request_ordinal]
  for name, _ in choice {
    if name != "index" &&
      name != "delta" &&
      name != "finish_reason" &&
      name != "logprobs" {
      raise OpenAIBaselineAmbiguousEvent
    }
  }
  let index = match choice.get("index") {
    Some(value) => openai_baseline_exact_int(value)
    None => raise OpenAIBaselineAmbiguousEvent
  }
  if index != 0L {
    raise OpenAIBaselineAmbiguousEvent
  }
  let delta = match choice.get("delta") {
    Some(Object(value)) => value
    _ => raise OpenAIBaselineAmbiguousEvent
  }
  let content = openai_baseline_delta_content(delta)
  let has_content = match content {
    Some(value) => !value.is_empty()
    None => false
  }
  let token_count = openai_baseline_choice_token_count(choice, has_content)
  match content {
    Some(value) if !value.is_empty() => {
      if lane.finish_seen {
        raise OpenAIBaselineAmbiguousEvent
      }
      if !lane.first_content_seen {
        self.collector.first_token(request_ordinal~) catch {
          _ => raise OpenAIBaselineCollectorRejected
        }
        lane.first_content_seen = true
      }
      if lane.observed_tokens > 9223372036854775807L - token_count {
        raise OpenAIBaselineInvalidUsage
      }
      lane.observed_tokens += token_count
    }
    _ => ()
  }
  match choice.get("finish_reason") {
    Some(Null) => ()
    Some(String(reason)) if reason == "stop" || reason == "length" => {
      if lane.finish_seen {
        raise OpenAIBaselineAmbiguousEvent
      }
      lane.finish_seen = true
    }
    _ => raise OpenAIBaselineAmbiguousEvent
  }
}

///|
fn OpenAIBaselineTrialAdapter::observe_json_event(
  self : OpenAIBaselineTrialAdapter,
  request_ordinal : Int,
  document : StringView,
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
  guard parsed is Object(root) else { raise OpenAIBaselineAmbiguousEvent }
  if root.contains("error") {
    raise OpenAIBaselineAmbiguousEvent
  }
  let choices = match root.get("choices") {
    Some(Array(values)) => values
    _ => raise OpenAIBaselineAmbiguousEvent
  }
  if choices.length() > 1 {
    raise OpenAIBaselineAmbiguousEvent
  }
  if choices.length() == 1 {
    guard choices[0] is Object(choice) else {
      raise OpenAIBaselineAmbiguousEvent
    }
    self.observe_choice(request_ordinal, choice)
  }
  match openai_baseline_optional_usage(root) {
    Some(usage) => self.observe_usage(request_ordinal, usage)
    None => ()
  }
  if choices.is_empty() && openai_baseline_optional_usage(root) is None {
    raise OpenAIBaselineAmbiguousEvent
  }
}
