///|
fn validate_path_identifier(
  value : StringView,
  file : ArtifactFileRef,
  maximum : Int,
) -> Unit raise KernelArtifactFileError {
  if value.is_empty() {
    raise InvalidPath(file~, issue=EmptyPath)
  }
  if value.length() > maximum {
    raise InvalidPath(file~, issue=PathTooLong)
  }
  if value[0] == '/' || value[0] == '\\' {
    raise InvalidPath(file~, issue=AbsolutePath)
  }
  let mut segment_length = 0
  let mut segment_dots = 0
  for unit in value {
    if unit == '/' {
      if segment_length == 0 ||
        (segment_length <= 2 && segment_dots == segment_length) {
        raise InvalidPath(file~, issue=TraversalSegment)
      }
      segment_length = 0
      segment_dots = 0
    } else {
      let valid = unit is ('A'..='Z' | 'a'..='z' | '0'..='9' | '_' | '-' | '.')
      if !valid {
        raise InvalidPath(file~, issue=InvalidPathCharacter)
      }
      segment_length += 1
      if unit == '.' {
        segment_dots += 1
      }
    }
  }
  if segment_length == 0 ||
    (segment_length <= 2 && segment_dots == segment_length) {
    raise InvalidPath(file~, issue=TraversalSegment)
  }
}
