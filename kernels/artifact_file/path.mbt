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

///|
fn join_path(root : StringView, identifier : StringView) -> String {
  if root.has_suffix("/") {
    "\{root}\{identifier}"
  } else {
    "\{root}/\{identifier}"
  }
}

///|
async fn validate_path_components(
  root : StringView,
  identifier : StringView,
  file : ArtifactFileRef,
) -> Unit raise KernelArtifactFileError {
  let mut relative = ""
  let components : Array[String] = []
  for component in identifier.split("/") {
    components.push(component.to_owned())
  }
  for index, component in components {
    relative = if relative.is_empty() {
      component
    } else {
      "\{relative}/\{component}"
    }
    let kind = @fs.kind(join_path(root, relative), follow_symlink=false) catch {
      _ => raise FileUnavailable(file)
    }
    if index + 1 < components.length() {
      if kind != Directory {
        raise UnsupportedPathType(file)
      }
    } else if kind != Regular {
      raise UnsupportedPathType(file)
    }
  }
}
