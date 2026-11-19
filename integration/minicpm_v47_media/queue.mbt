///|
pub(all) struct PreparedRequest {
  id : String
  input : Prepared
  maximum_new_tokens : Int32
  context_ceiling : Int32
}

///|
fn positive_integer(value : Json, maximum : Int32) -> Int32 raise {
  guard value is Number(number, ..) &&
    number > 0.0 &&
    number <= maximum.to_double() &&
    number == number.to_int().to_double() else {
    fail("invalid media request integer")
  }
  number.to_int()
}

///|
/// Local queue DTO. The caller resolves image paths into decoded RGB frames;
/// this layer has no filesystem/URL authority or vendor device dependency.
pub fn prepare_requests(
  spec : @model.Spec,
  tokenizer : @tokenizer.TokenizerSpec,
  vision : @vision.Program,
  document : Json,
  load_image~ : (String) -> @rgb8.Image raise,
  maximum_prepared_pixel_bytes~ : Int64,
  maximum_image_working_bytes~ : Int64 = 268435456L,
) -> ReadOnlyArray[PreparedRequest] raise {
  guard document is Object(envelope) &&
    envelope.length() == 2 &&
    envelope.get("format") ==
    Some(Json("lunaflux.minicpm-v47-media-requests.v1")) &&
    envelope.get("requests") is Some(Array(requests)) &&
    !requests.is_empty() &&
    requests.length() <= 32 &&
    maximum_prepared_pixel_bytes >= 0L else {
    fail("invalid media request queue")
  }
  let prepared : Array[PreparedRequest] = []
  let ids : Map[String, Unit] = Map([])
  let mut remaining = maximum_prepared_pixel_bytes
  for request in requests {
    guard request is Object(fields) &&
      fields.get("id") is Some(String(id)) &&
      !id.is_empty() &&
      id.length() <= 128 &&
      !ids.contains(id) &&
      fields.get("system") is Some(String(system)) &&
      fields.get("user") is Some(Array(parts)) &&
      !parts.is_empty() &&
      parts.length() <= 64 &&
      fields.get("maximum_new_tokens") is Some(new_tokens) else {
      fail("invalid media request")
    }
    for field, _ in fields {
      guard field
        is ("id" | "system" | "user" | "maximum_new_tokens" | "context_ceiling") else {
        fail("unsupported media generation option")
      }
    }
    ids[id] = ()
    let maximum_new_tokens = positive_integer(new_tokens, 4096)
    let context_ceiling = match fields.get("context_ceiling") {
      Some(value) => positive_integer(value, 4096)
      None => 4096
    }
    let content : Array[Content] = []
    let mut image_count = 0
    for part in parts {
      guard part is Object(fields) && fields.length() == 2 else {
        fail("invalid media content part")
      }
      match (fields.get("type"), fields.get("text"), fields.get("path")) {
        (Some(String("text")), Some(String(text)), None) =>
          content.push(Text(text))
        (Some(String("image")), None, Some(String(path))) => {
          guard image_count < 8 && !path.is_empty() else {
            fail("invalid media image part")
          }
          image_count += 1
          content.push(Image(load_image(path)))
        }
        _ => fail("unsupported media content part")
      }
    }
    let input = prepare(
      spec,
      tokenizer,
      vision,
      system,
      content[:],
      maximum_tokens=context_ceiling,
      maximum_prepared_pixel_bytes=remaining,
      maximum_image_working_bytes~,
    )
    remaining -= input.pixel_bytes()
    prepared.push({ id, input, maximum_new_tokens, context_ceiling, })
  }
  ReadOnlyArray::from_array(prepared)
}
