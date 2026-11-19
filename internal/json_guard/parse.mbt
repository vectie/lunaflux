///|
/// Parse JSON with a caller-owned nesting limit, without the deprecated parser
/// option. The lexical preflight ignores delimiters inside strings; the core
/// parser still validates syntax and retains its duplicate-key behavior.
/// Limits cannot exceed the core parser's default depth of 1024.
pub fn parse_bounded(input : StringView, max_depth~ : Int32) -> Json raise {
  guard max_depth > 0 && max_depth <= 1024 else { raise InvalidDepthLimit }
  let mut depth = 0
  let mut in_string = false
  let mut escaped = false
  for unit in input {
    if in_string {
      if escaped {
        escaped = false
      } else if unit == '\\' {
        escaped = true
      } else if unit == '"' {
        in_string = false
      }
    } else {
      match unit {
        '"' => in_string = true
        '{' | '[' => {
          depth += 1
          guard depth <= max_depth else {
            raise DepthLimitExceeded(limit=max_depth)
          }
        }
        '}' | ']' => depth -= 1
        _ => ()
      }
    }
  }
  @json.parse(input)
}
