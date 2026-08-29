///|
/// Parse a bounded UTF-8 tokenizer.json byte sequence.
pub fn parse_bytes(
  document : BytesView,
  digest~ : @tokenizer.TokenizerDigest,
  limits~ : @tokenizer.TokenizerLimits,
  max_bytes~ : Int,
  max_depth~ : Int,
) -> @tokenizer.TokenizerSpec raise TokenizerJsonError {
  validate_reader_limits(max_bytes, max_depth)
  if document.length() > max_bytes {
    raise ByteLimitExceeded(limit=max_bytes)
  }
  let text = @utf8.decode(document) catch { _ => raise InvalidUtf8 }
  parse_text(text, digest, limits, max_depth)
}

///|
/// Parse tokenizer.json text after bounding its UTF-8 representation.
pub fn parse_string(
  document : StringView,
  digest~ : @tokenizer.TokenizerDigest,
  limits~ : @tokenizer.TokenizerLimits,
  max_bytes~ : Int,
  max_depth~ : Int,
) -> @tokenizer.TokenizerSpec raise TokenizerJsonError {
  validate_reader_limits(max_bytes, max_depth)
  validate_utf8_size(document, max_bytes)
  parse_text(document, digest, limits, max_depth)
}

///|
fn parse_text(
  document : StringView,
  digest : @tokenizer.TokenizerDigest,
  limits : @tokenizer.TokenizerLimits,
  max_depth : Int,
) -> @tokenizer.TokenizerSpec raise TokenizerJsonError {
  @json_guard.reject_duplicate_object_keys(document, max_depth~) catch {
    InvalidDepthLimit => raise InvalidDepthLimit
    DepthLimitExceeded(limit~) => raise NestingLimitExceeded(limit~)
    DuplicateObjectKey(depth~, index~) => raise DuplicateField(depth~, index~)
    MalformedStructure => raise MalformedJson
  }
  let parsed = @json.parse(document) catch { _ => raise MalformedJson }
  guard parsed is Object(root) else { raise NonObjectRoot }
  let profile = validate_root_semantics(root)
  let specials = parse_added_tokens(root, limits, profile)
  parse_model(
    require_field(root, "model", Root, ModelField),
    specials,
    digest,
    limits,
    profile,
  )
}
