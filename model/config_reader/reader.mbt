///|
fn validate_limits(
  max_bytes : Int,
  max_depth : Int,
) -> Unit raise ModelConfigReaderError {
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
) -> Unit raise ModelConfigReaderError {
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
  plan_digest : @spec.PlanDigest,
  max_depth : Int,
) -> @spec.LlamaModelMetadata raise ModelConfigReaderError {
  @json_guard.reject_duplicate_object_keys(document, max_depth~) catch {
    InvalidDepthLimit => raise InvalidDepthLimit
    DepthLimitExceeded(limit~) => raise NestingLimitExceeded(limit~)
    DuplicateObjectKey(depth~, index~) => raise DuplicateField(depth~, index~)
    MalformedStructure => raise MalformedJson
  }
  let parsed = @json.parse(document) catch { _ => raise MalformedJson }
  guard parsed is Object(root) else { raise NonObjectRoot }
  decode_model(root, content_digest, plan_digest)
}

///|
/// Parses selected-model JSON bytes after enforcing byte and nesting bounds.
///
/// The supplied digests must already have been validated from independently
/// verified artifact content and the canonical plan. This function performs
/// no file access, hashing, code loading, or weight materialization.
pub fn parse_bytes(
  document : BytesView,
  content_digest : @spec.ContentDigest,
  plan_digest : @spec.PlanDigest,
  max_bytes~ : Int,
  max_depth~ : Int,
) -> @spec.LlamaModelMetadata raise ModelConfigReaderError {
  validate_limits(max_bytes, max_depth)
  if document.length() > max_bytes {
    raise ByteLimitExceeded(limit=max_bytes)
  }
  let text = @utf8.decode(document) catch { _ => raise InvalidUtf8 }
  parse_text(text, content_digest, plan_digest, max_depth)
}

///|
/// Parses selected-model JSON text under its encoded UTF-8 byte bound.
///
/// Invalid UTF-16 structure is rejected while measuring so no encoded copy is
/// allocated merely to enforce `max_bytes`.
pub fn parse_string(
  document : StringView,
  content_digest : @spec.ContentDigest,
  plan_digest : @spec.PlanDigest,
  max_bytes~ : Int,
  max_depth~ : Int,
) -> @spec.LlamaModelMetadata raise ModelConfigReaderError {
  validate_limits(max_bytes, max_depth)
  validate_utf8_size(document, max_bytes)
  parse_text(document, content_digest, plan_digest, max_depth)
}
