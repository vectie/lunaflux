///|
pub struct NormalizedChat {
  priv body : Bytes
  priv system : String
  priv system_present : Bool
  priv content : ReadOnlyArray[Content]
}

///|
pub fn NormalizedChat::body(self : Self) -> Bytes {
  self.body
}

///|
pub fn NormalizedChat::system(self : Self) -> String {
  self.system
}

///|
pub fn NormalizedChat::content(self : Self) -> ReadOnlyArray[Content] {
  self.content
}

///|
pub fn NormalizedChat::system_present(self : Self) -> Bool {
  self.system_present
}

///|
fn decode_image_url(url : String) -> @rgb8.Image raise {
  let (payload, png) = if url.strip_prefix("data:image/png;base64,")
    is Some(payload) {
    (payload, true)
  } else if url.strip_prefix("data:image/jpeg;base64,") is Some(payload) {
    (payload, false)
  } else {
    fail("image_url requires an inline PNG/JPEG data URI")
  }
  guard payload.length() > 0 && payload.length() <= 44739244 else {
    fail("encoded image exceeds media limit")
  }
  let bytes = @base64.decode(payload, ignore_whitespace=false)
  guard (if png {
    bytes.has_prefix(b"\x89PNG\r\n\x1a\n")
  } else {
    bytes.has_prefix(b"\xff\xd8")
  }) else {
    fail("image MIME and encoding differ")
  }
  @rgb8.Image::decode_oriented(
    bytes,
    maximum_encoded_bytes=33554432,
    maximum_pixels=16777216,
  )
}

///|
/// Normalize stateless OpenAI text/image content for the existing canonical
/// text admission parser. Retains all generation fields for that parser to
/// validate; image data becomes native placeholders plus a typed RGB sidecar.
/// The caller already owns authenticated, bounded HTTP body bytes.
/// `allow_images=false` retains the same text/list normalization and rejects
/// image parts before decoding, for a resident with no vision program.
pub fn normalize_chat(
  spec : @model.Spec,
  body : Bytes,
  allow_images~ : Bool = true,
) -> NormalizedChat raise {
  guard !body.is_empty() && body.length() <= 33554432 else {
    fail("invalid media HTTP body")
  }
  let text = @utf8.decode(body)
  @json_guard.reject_duplicate_object_keys(text[:], max_depth=16)
  guard @json.parse(text, max_nesting_depth=16) is Object(fields) &&
    fields.get("messages") is Some(Array(messages)) &&
    messages.length() > 0 &&
    messages.length() <= 2 else {
    fail("invalid media chat messages")
  }
  let system = if messages.length() == 2 {
    guard messages[0] is Object(message) &&
      message.length() == 2 &&
      message.get("role") == Some(Json("system")) &&
      message.get("content") is Some(String(system)) else {
      fail("media chat requires optional system then user")
    }
    system
  } else {
    ""
  }
  guard messages[messages.length() - 1] is Object(user) &&
    user.length() == 2 &&
    user.get("role") == Some(Json("user")) &&
    user.get("content") is Some(value) else {
    fail("media chat requires one final user message")
  }
  let parts : Array[Json] = match value {
    String(text) => [{ "type": "text", "text": Json(text) }]
    Array(parts) => parts
    _ => fail("invalid media content")
  }
  guard !parts.is_empty() && parts.length() <= 64 else {
    fail("invalid media content parts")
  }
  let content : Array[Content] = []
  let rendered : Array[String] = []
  let mut images = 0
  for part in parts {
    guard part is Object(part) && part.length() == 2 else {
      fail("invalid media content part")
    }
    match (part.get("type"), part.get("text"), part.get("image_url")) {
      (Some(String("text")), Some(String(text)), None) => {
        content.push(Text(text))
        rendered.push(text)
      }
      (Some(String("image_url")), None, Some(Object(image))) => {
        guard allow_images &&
          images < 8 &&
          image.length() >= 1 &&
          image.length() <= 2 &&
          image.get("url") is Some(String(url)) else {
          fail("invalid image_url part")
        }
        for key, _ in image {
          guard key is ("url" | "detail") else {
            fail("unsupported image option")
          }
        }
        guard image.get("detail") is (None | Some(String("auto" | "high"))) else {
          fail("unsupported image detail")
        }
        let image = decode_image_url(url)
        let plan = @processor.Plan::build(
          spec,
          height=image.height(),
          width=image.width(),
        )
        rendered.push(plan.placeholder_text(image_id=images))
        content.push(Image(image))
        images += 1
      }
      _ => fail("unsupported media content part")
    }
  }
  let normalized : Array[Json] = []
  if messages.length() == 2 {
    normalized.push({
      "role": "system",
      "content": Json(@prompt.trim_message_content(system)),
    })
  }
  normalized.push({
    "role": "user",
    "content": Json(@prompt.trim_message_content(rendered.join("\n"))),
  })
  fields["messages"] = Json(normalized)
  let body = @utf8.encode(Json(fields).stringify())
  guard body.length() <= 1048576 else {
    fail("normalized media chat exceeds text admission limit")
  }
  {
    body,
    system,
    system_present: messages.length() == 2,
    content: ReadOnlyArray::from_array(content),
  }
}
