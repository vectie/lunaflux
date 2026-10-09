///|
fn validate_limits(
  max_bytes : Int,
  max_depth : Int,
) -> Unit raise Glm53ConfigError {
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
) -> Unit raise Glm53ConfigError {
  let mut index = 0
  let mut bytes = 0
  while index < input.length() {
    let unit = input[index]
    let (width, next) = if unit <= 0x7f {
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
    if bytes > max_bytes - width {
      raise ByteLimitExceeded(limit=max_bytes)
    }
    bytes += width
    index = next
  }
}

///|
fn parse_root(
  document : StringView,
  max_depth : Int,
) -> Map[String, Json] raise Glm53ConfigError {
  @json_guard.reject_duplicate_object_keys(document, max_depth~) catch {
    InvalidDepthLimit => raise InvalidDepthLimit
    DepthLimitExceeded(limit~) => raise NestingLimitExceeded(limit~)
    DuplicateObjectKey(depth~, index~) => raise DuplicateField(depth~, index~)
    MalformedStructure => raise MalformedJson
  }
  let parsed = @json.parse(document) catch { _ => raise MalformedJson }
  guard parsed is Object(root) else { raise NonObjectRoot }
  root
}

///|
pub fn parse_glm53_config(
  document : StringView,
  content_digest : @spec.ContentDigest,
  max_bytes~ : Int,
  max_depth~ : Int,
) -> Glm53ConfigAdmission raise Glm53ConfigError {
  validate_limits(max_bytes, max_depth)
  validate_utf8_size(document, max_bytes)
  let root = parse_root(document, max_depth)
  match require_string(root, "model_type", ModelType) {
    "glm_moe_dsa" => {
      let spec = validate_full_config(root)
      Full(
        @glm53_spec.Glm53FullMetadata::from_verified_content(
          spec, content_digest,
        ),
      )
    }
    "glm5_next" => {
      let spec = validate_flash_config(root)
      Flash(
        @glm53_spec.Glm53FlashMetadata::from_verified_content(
          spec, content_digest,
        ),
      )
    }
    _ => raise invalid(ModelType)
  }
}

///|
pub fn parse_glm53_config_bytes(
  document : Bytes,
  content_digest : @spec.ContentDigest,
  max_bytes~ : Int,
  max_depth~ : Int,
) -> Glm53ConfigAdmission raise Glm53ConfigError {
  validate_limits(max_bytes, max_depth)
  if document.length() > max_bytes {
    raise ByteLimitExceeded(limit=max_bytes)
  }
  let text = @utf8.decode(document) catch { _ => raise InvalidUtf8 }
  parse_glm53_config(text, content_digest, max_bytes~, max_depth~)
}
