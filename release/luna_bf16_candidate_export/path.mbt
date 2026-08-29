///|
fn output_path_unit_is_safe(unit : Char) -> Bool {
  unit is ('A'..='Z' | 'a'..='z' | '0'..='9' | '.' | '_' | '-' | '/')
}

///|
fn validate_new_output_path(
  path : String,
) -> (String, String) raise LunaBf16CandidateExportError {
  if path.length() < 2 ||
    path.length() > 4096 ||
    path[0] != '/' ||
    path[path.length() - 1] == '/' ||
    path.contains("//") ||
    path.contains("/./") ||
    path.contains("/../") ||
    path.has_suffix("/.") ||
    path.has_suffix("/..") {
    raise ExportFailed(OutputPath)
  }
  for unit in path {
    if !output_path_unit_is_safe(unit) {
      raise ExportFailed(OutputPath)
    }
  }
  guard path.rev_find("/") is Some(separator) && separator < path.length() - 1 else {
    raise ExportFailed(OutputPath)
  }
  let parent = if separator == 0 { "/" } else { path[:separator].to_owned() }
  let basename = path[separator + 1:].to_owned()
  if basename == "." || basename == ".." {
    raise ExportFailed(OutputPath)
  }
  (parent, basename)
}

///|
async fn admit_new_output_path(
  path : String,
) -> String raise LunaBf16CandidateExportError {
  let (parent, basename) = validate_new_output_path(path)
  let actual_parent = @fs.realpath(parent) catch {
    _ => raise ExportFailed(OutputPath)
  }
  let output_exists = @fs.exists(path) catch {
    _ => raise ExportFailed(OutputPath)
  }
  if actual_parent != parent || output_exists {
    raise ExportFailed(OutputPath)
  }
  let staging = if parent == "/" {
    "/." + basename + ".lunaflux-staging"
  } else {
    parent + "/." + basename + ".lunaflux-staging"
  }
  let staging_exists = @fs.exists(staging) catch {
    _ => raise ExportFailed(OutputPath)
  }
  if staging_exists {
    raise ExportFailed(OutputPath)
  }
  staging
}
