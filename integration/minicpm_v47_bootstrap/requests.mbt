///|
pub(all) struct PreparedRequest {
  id : String
  prompt : @prompt.PreparedTextPrompt
  maximum_new_tokens : Int32
  context_ceiling : Int32
}

///|
fn positive_integer(value : Json, maximum : Int32) -> Int32 raise {
  guard value is Number(number, ..) &&
    number > 0.0 &&
    number <= maximum.to_double() &&
    number == number.to_int().to_double() else {
    fail("invalid request integer")
  }
  number.to_int()
}

///|
/// Initial explicit greedy text queue. Unsupported generation/media options are
/// rejected rather than silently discarded. Tokenization runs before CUDA.
pub fn prepare_requests(
  model : @model.Spec,
  tokenizer : @tokenizer.TokenizerSpec,
  document : Json,
) -> ReadOnlyArray[PreparedRequest] raise {
  guard document is Object(envelope) &&
    envelope.length() == 2 &&
    envelope.get("format") ==
    Some(Json("lunaflux.minicpm-v47-text-requests.v1")) &&
    envelope.get("requests") is Some(Array(requests)) &&
    requests.length() > 0 &&
    requests.length() <= 32 else {
    fail("invalid text request queue")
  }
  let prepared = []
  let ids : Map[String, Unit] = Map([])
  for request in requests {
    guard request is Object(fields) &&
      fields.get("id") is Some(String(id)) &&
      !id.is_empty() &&
      id.length() <= 128 &&
      fields.get("system") is Some(String(system)) &&
      fields.get("user") is Some(String(user)) &&
      fields.get("maximum_new_tokens") is Some(new_tokens) else {
      fail("invalid text request")
    }
    for field, _ in fields {
      guard field
        is ("id" | "system" | "user" | "maximum_new_tokens" | "context_ceiling") else {
        fail("unsupported text request option")
      }
    }
    guard !ids.contains(id) else { fail("duplicate text request ID") }
    ids[id] = ()
    let maximum_new_tokens = positive_integer(new_tokens, 4096)
    let context_ceiling = match fields.get("context_ceiling") {
      Some(value) => positive_integer(value, 4096)
      None => 4096
    }
    let prompt = @prompt.prepare_text_prompt(
      model,
      tokenizer,
      system,
      user,
      maximum_tokens=context_ceiling,
    )
    prepared.push({ id, prompt, maximum_new_tokens, context_ceiling, })
  }
  ReadOnlyArray::from_array(prepared)
}
