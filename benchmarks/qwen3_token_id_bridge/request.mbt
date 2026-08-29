///|
fn bridge_exact_fields(
  object : Map[String, Json],
  fields : ArrayView[String],
) -> Bool {
  if object.length() != fields.length() {
    return false
  }
  for field in fields {
    if !object.contains(field) {
      return false
    }
  }
  true
}

///|
fn bridge_exact_int(
  value : Json,
  lower : Int,
  upper : Int,
) -> Int raise Qwen3TokenBridgeError {
  guard value is Number(number, repr=_) else { raise MalformedRequest }
  let parsed = number.to_int()
  if number != number.floor() || parsed < lower || parsed > upper {
    raise UnsupportedRequest
  }
  parsed
}

///|
pub fn parse_qwen3_token_id_request(
  document : StringView,
  max_input_tokens : Int,
  max_output_tokens : Int,
  max_token_id : Int,
) -> Qwen3TokenIdRequest raise Qwen3TokenBridgeError {
  if document.length() == 0 ||
    document.length() > 131072 ||
    max_input_tokens <= 0 ||
    max_output_tokens <= 0 ||
    max_token_id <= 0 {
    raise RequestTooLarge
  }
  @json_guard.reject_duplicate_object_keys(document, max_depth=4) catch {
    _ => raise MalformedRequest
  }
  let parsed = @json.parse(document) catch { _ => raise MalformedRequest }
  guard parsed is Object(root) &&
    bridge_exact_fields(root, [
      "schema", "model", "input_token_ids", "max_output_tokens", "sampling", "stream",
    ]) &&
    root.get("schema")
    is Some(String("lunaflux.benchmark-token-ids-request.v1")) &&
    root.get("model") is Some(String("Qwen3-0.6B")) &&
    root.get("stream") is Some(True) &&
    root.get("input_token_ids") is Some(Array(input_values)) &&
    root.get("max_output_tokens") is Some(output_value) &&
    root.get("sampling") is Some(Object(sampling)) else {
    raise UnsupportedRequest
  }
  if input_values.is_empty() ||
    input_values.length() > max_input_tokens ||
    !bridge_exact_fields(sampling, ["mode", "temperature", "top_p", "seed"]) ||
    !(sampling.get("mode") is Some(String("greedy"))) ||
    !(sampling.get("temperature") is Some(Number(0.0, ..))) ||
    !(sampling.get("top_p") is Some(Number(1.0, ..))) ||
    !(sampling.get("seed") is Some(Number(0.0, ..))) {
    raise UnsupportedRequest
  }
  let ids : Array[Int] = []
  for value in input_values {
    ids.push(bridge_exact_int(value, 0, max_token_id))
  }
  let output = bridge_exact_int(output_value, 1, max_output_tokens)
  { input_token_ids: ids, max_output_tokens: output }
}
