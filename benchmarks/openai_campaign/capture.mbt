///|
fn campaign_hex_nibble(
  unit : UInt16,
) -> Byte raise OpenAIResponsesCampaignError {
  match unit {
    '0'..='9' => (unit.to_int() - '0'.to_int()).to_byte()
    'a'..='f' => (unit.to_int() - 'a'.to_int() + 10).to_byte()
    _ => raise CampaignInvalidHex
  }
}

///|
fn campaign_decode_hex(
  value : String,
) -> Bytes raise OpenAIResponsesCampaignError {
  if value.is_empty() || value.length() % 2 != 0 || value.length() > 2097152 {
    raise CampaignInvalidHex
  }
  Bytes::makei(value.length() / 2, index => {
    let high = campaign_hex_nibble(value[index * 2])
    let low = campaign_hex_nibble(value[index * 2 + 1])
    (high.to_int() * 16 + low.to_int()).to_byte()
  })
}

///|
fn campaign_transport_outcome(
  value : String,
) -> @evidence.BenchmarkRequestOutcome raise OpenAIResponsesCampaignError {
  match value {
    "timed_out" => TimedOut
    "cancelled" => Cancelled
    "failed" => Failed
    _ => raise CampaignInvalidCapture
  }
}

///|
fn replay_campaign_event(
  observer : @openai_adapter.OpenAIResponsesTrialObserver,
  event : Json,
  tokens : FixedArray[Int64],
) -> Unit raise OpenAIResponsesCampaignError {
  let object = campaign_require_object(event, 8)
  let kind = campaign_string(object, "kind")
  let request = campaign_decimal_int(
    campaign_string(object, "request_ordinal"),
    true,
  )
  if request < 0 || request >= tokens.length() {
    raise CampaignInvalidCapture
  }
  let timestamp = campaign_timestamp(campaign_string(object, "timestamp_ns"))
  let input_tokens = campaign_decimal_i64(
    campaign_string(object, "input_tokens"),
    true,
  )
  let status = campaign_decimal_int(campaign_string(object, "status"), true)
  let content_type = campaign_string(object, "content_type")
  let outcome = campaign_string(object, "outcome")
  let bytes_hex = campaign_string(object, "bytes_hex")
  match kind {
    "submit" => {
      if input_tokens != tokens[request] ||
        status != 0 ||
        !content_type.is_empty() ||
        !outcome.is_empty() ||
        !bytes_hex.is_empty() {
        raise CampaignInvalidCapture
      }
      observer.submit_captured(request, timestamp) catch {
        _ => raise CampaignAdapterRejected
      }
    }
    "response_head" => {
      if input_tokens != 0L ||
        status < 100 ||
        status > 599 ||
        content_type.is_empty() ||
        !outcome.is_empty() ||
        !bytes_hex.is_empty() {
        raise CampaignInvalidCapture
      }
      observer.observe_response_head_captured(
        request, status, content_type, timestamp,
      ) catch {
        _ => raise CampaignAdapterRejected
      }
    }
    "sse_bytes" => {
      if input_tokens != 0L ||
        status != 0 ||
        !content_type.is_empty() ||
        !outcome.is_empty() {
        raise CampaignInvalidCapture
      }
      observer.offer_sse_bytes_captured(
        request,
        campaign_decode_hex(bytes_hex),
        timestamp,
      ) catch {
        _ => raise CampaignAdapterRejected
      }
    }
    "transport_terminal" => {
      if input_tokens != 0L ||
        status != 0 ||
        !content_type.is_empty() ||
        outcome.is_empty() ||
        !bytes_hex.is_empty() {
        raise CampaignInvalidCapture
      }
      observer.terminate_transport_captured(
        request,
        campaign_transport_outcome(outcome),
        timestamp,
      ) catch {
        _ => raise CampaignAdapterRejected
      }
    }
    _ => raise CampaignInvalidCapture
  }
}

///|
fn replay_campaign_trial(
  declaration : OpenAIResponsesCampaignDeclaration,
  expected_index : Int,
  bytes : BytesView,
) -> (@openai_adapter.OpenAIResponsesTrialEvidence, @evidence.BenchmarkDigest) raise OpenAIResponsesCampaignError {
  let root = campaign_require_object(campaign_parse_json(bytes, 8), 8)
  if campaign_string(root, "schema") !=
    "lunaflux-openai-responses-captured-trial.v1" ||
    campaign_string(root, "declaration_sha256") != declaration.digest.as_hex() {
    raise CampaignInvalidSchema
  }
  let index = campaign_decimal_int(campaign_string(root, "trial_index"), true)
  if index != expected_index {
    raise CampaignInvalidShape
  }
  let profile_index = index / 9
  let ordinal = index % 9 / 3 + 1
  let engine_index = index % 3
  let position = campaign_order_position(engine_index, ordinal)
  let profile = declaration.profiles[profile_index]
  let engine = declaration.engines[engine_index]
  if campaign_profile(campaign_string(root, "profile")) != profile.profile ||
    campaign_engine(campaign_string(root, "engine")) != engine.engine ||
    campaign_decimal_int(campaign_string(root, "ordinal"), false) != ordinal ||
    campaign_decimal_int(campaign_string(root, "order_position"), true) !=
    position {
    raise CampaignInvalidShape
  }
  let contract = @openai_adapter.OpenAIResponsesTrialContract::new(
    engine.identity,
    @openai_adapter.OpenAIBaselineInputs::new(
      profile.profile,
      declaration.tokenizer,
      declaration.corpus,
      declaration.protocol,
      profile.workload,
      profile.request_input_tokens,
    ) catch {
      _ => raise CampaignAdapterRejected
    },
    ordinal,
    position,
    1048576,
    16,
  ) catch {
    _ => raise CampaignAdapterRejected
  }
  let observer = contract.start_captured() catch {
    _ => raise CampaignAdapterRejected
  }
  let events = campaign_array(root, "events")
  if events.is_empty() || events.length() > 1048576 {
    raise CampaignInvalidCapture
  }
  for event in events {
    replay_campaign_event(observer, event, profile.request_input_tokens)
  }
  let evidence = observer.finish() catch { _ => raise CampaignAdapterRejected }
  (evidence, campaign_sha256(bytes))
}
