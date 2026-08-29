///|
fn campaign_sha256(bytes : BytesView) -> @evidence.BenchmarkDigest {
  try! @evidence.BenchmarkDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
}

///|
fn campaign_parse_json(
  bytes : BytesView,
  maximum_depth : Int,
) -> Json raise OpenAIResponsesCampaignError {
  let document = @utf8.decode(bytes) catch { _ => raise CampaignInvalidJson }
  @json_guard.reject_duplicate_object_keys(document, max_depth=maximum_depth) catch {
    _ => raise CampaignInvalidJson
  }
  @json.parse(document) catch {
    _ => raise CampaignInvalidJson
  }
}

///|
fn campaign_require_object(
  value : Json,
  fields : Int,
) -> Map[String, Json] raise OpenAIResponsesCampaignError {
  guard value is Object(object) && object.length() == fields else {
    raise CampaignInvalidJson
  }
  object
}

///|
fn campaign_string(
  object : Map[String, Json],
  field : String,
) -> String raise OpenAIResponsesCampaignError {
  match object.get(field) {
    Some(String(value)) => value
    _ => raise CampaignInvalidJson
  }
}

///|
fn campaign_array(
  object : Map[String, Json],
  field : String,
) -> Array[Json] raise OpenAIResponsesCampaignError {
  match object.get(field) {
    Some(Array(values)) => values
    _ => raise CampaignInvalidJson
  }
}

///|
fn campaign_decimal_i64(
  value : String,
  allow_zero : Bool,
) -> Int64 raise OpenAIResponsesCampaignError {
  if value.is_empty() || value.length() > 19 {
    raise CampaignInvalidJson
  }
  let mut result = 0L
  for index, unit in value {
    guard unit is ('0'..='9') else { raise CampaignInvalidJson }
    if index == 0 && value.length() > 1 && unit == '0' {
      raise CampaignInvalidJson
    }
    let digit = (unit.to_int() - '0'.to_int()).to_int64()
    if result > (9223372036854775807L - digit) / 10L {
      raise CampaignInvalidJson
    }
    result = result * 10L + digit
  }
  if !allow_zero && result == 0L {
    raise CampaignInvalidJson
  }
  result
}

///|
fn campaign_decimal_int(
  value : String,
  allow_zero : Bool,
) -> Int raise OpenAIResponsesCampaignError {
  let parsed = campaign_decimal_i64(value, allow_zero)
  if parsed > 2147483647L {
    raise CampaignInvalidJson
  }
  parsed.to_int()
}

///|
fn campaign_digest(
  value : String,
) -> @evidence.BenchmarkDigest raise OpenAIResponsesCampaignError {
  @evidence.BenchmarkDigest::from_sha256(value) catch {
    _ => raise CampaignInvalidDigest
  }
}

///|
fn campaign_timestamp(
  value : String,
) -> @runner.CapturedMonotonicTimestamp raise OpenAIResponsesCampaignError {
  @runner.CapturedMonotonicTimestamp::from_nanoseconds(
    campaign_decimal_i64(value, true),
  ) catch {
    _ => raise CampaignInvalidTimestamp
  }
}

///|
fn campaign_engine(
  value : String,
) -> @evidence.BenchmarkEngine raise OpenAIResponsesCampaignError {
  match value {
    "lunaflux" => LunaFlux
    "vllm" => Vllm
    "sglang" => Sglang
    _ => raise CampaignInvalidDeclaration
  }
}

///|
fn campaign_profile(
  value : String,
) -> @evidence.BenchmarkProfile raise OpenAIResponsesCampaignError {
  match value {
    "latency" => Latency
    "chat" => Chat
    "long_prefill" => LongPrefill
    "decode_heavy" => DecodeHeavy
    "prefix_rich" => PrefixRich
    "prefix_cold" => PrefixCold
    "saturation" => Saturation
    "churn" => Churn
    "mixed" => Mixed
    _ => raise CampaignInvalidDeclaration
  }
}

///|
fn campaign_engine_at(index : Int) -> @evidence.BenchmarkEngine {
  match index {
    0 => LunaFlux
    1 => Vllm
    _ => Sglang
  }
}

///|
fn campaign_profile_at(index : Int) -> @evidence.BenchmarkProfile {
  match index {
    0 => Latency
    1 => Chat
    2 => LongPrefill
    3 => DecodeHeavy
    4 => PrefixRich
    5 => PrefixCold
    6 => Saturation
    7 => Churn
    _ => Mixed
  }
}

///|
fn campaign_order_position(engine_index : Int, ordinal : Int) -> Int {
  match ordinal {
    1 => engine_index
    2 => (engine_index + 2) % 3
    _ => (engine_index + 1) % 3
  }
}

///|
fn campaign_loopback_endpoint(
  value : String,
) -> Unit raise OpenAIResponsesCampaignError {
  guard value.split_once(":") is Some((host, port)) && host == "127.0.0.1" else {
    raise CampaignInvalidEndpoint
  }
  let parsed = campaign_decimal_int(port.to_owned(), false)
  if parsed > 65535 {
    raise CampaignInvalidEndpoint
  }
}
