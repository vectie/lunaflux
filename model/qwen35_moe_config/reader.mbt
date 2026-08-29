///|
fn validate_limits(
  max_bytes : Int,
  max_depth : Int,
) -> Unit raise Qwen35ConfigError {
  if max_bytes <= 0 {
    raise InvalidByteLimit
  }
  if max_depth <= 0 {
    raise InvalidDepthLimit
  }
}

///|
fn validate_utf8_size(
  input : StringView,
  max_bytes : Int,
) -> Unit raise Qwen35ConfigError {
  let mut index = 0
  let mut byte_count = 0
  while index < input.length() {
    let unit = input[index]
    let (width, next_index) = if unit <= 0x7f {
      (1, index + 1)
    } else if unit <= 0x7ff {
      (2, index + 1)
    } else if unit.is_leading_surrogate() {
      if index + 1 >= input.length() ||
        !input[index + 1].is_trailing_surrogate() {
        raise InvalidUtf8
      }
      (4, index + 2)
    } else if unit.is_trailing_surrogate() {
      raise InvalidUtf8
    } else {
      (3, index + 1)
    }
    if byte_count > max_bytes - width {
      raise ByteLimitExceeded(limit=max_bytes)
    }
    byte_count += width
    index = next_index
  }
}

///|
fn parse_text(
  document : StringView,
  content_digest : @spec.ContentDigest,
  max_depth : Int,
) -> @qwen35_moe_spec.Qwen35MoeTextMetadata raise Qwen35ConfigError {
  @json_guard.reject_duplicate_object_keys(document, max_depth~) catch {
    InvalidDepthLimit => raise InvalidDepthLimit
    DepthLimitExceeded(limit~) => raise NestingLimitExceeded(limit~)
    DuplicateObjectKey(depth~, index~) => raise DuplicateField(depth~, index~)
    MalformedStructure => raise MalformedJson
  }
  let parsed = @json.parse(document) catch { _ => raise MalformedJson }
  guard parsed is Object(root) else { raise NonObjectRoot }
  decode_root(root, content_digest)
}

///|
/// Parses the exact nested Qwen3.5-MoE config as a text-only, no-MTP profile.
///
/// The supplied digest must already bind the immutable config content. This
/// startup-only parser performs no file access, tensor admission, or execution
/// dispatch and never falls back to dense Qwen3 or another model family.
pub fn parse_string(
  document : StringView,
  content_digest : @spec.ContentDigest,
  max_bytes~ : Int,
  max_depth~ : Int,
) -> @qwen35_moe_spec.Qwen35MoeTextMetadata raise Qwen35ConfigError {
  validate_limits(max_bytes, max_depth)
  validate_utf8_size(document, max_bytes)
  parse_text(document, content_digest, max_depth)
}

///|
/// Byte-oriented form of `parse_string` with the same bounded semantics.
pub fn parse_bytes(
  document : BytesView,
  content_digest : @spec.ContentDigest,
  max_bytes~ : Int,
  max_depth~ : Int,
) -> @qwen35_moe_spec.Qwen35MoeTextMetadata raise Qwen35ConfigError {
  validate_limits(max_bytes, max_depth)
  if document.length() > max_bytes {
    raise ByteLimitExceeded(limit=max_bytes)
  }
  let text = @utf8.decode(document) catch { _ => raise InvalidUtf8 }
  parse_text(text, content_digest, max_depth)
}
