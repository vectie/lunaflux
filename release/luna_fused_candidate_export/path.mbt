///|
fn fused_export_path_unit_is_safe(unit : Char) -> Bool {
  unit is ('A'..='Z' | 'a'..='z' | '0'..='9' | '.' | '_' | '-' | '/')
}

///|
async fn fused_export_staging_path(
  path : String,
) -> String raise LunaFusedCandidateExportError {
  if path.length() < 2 ||
    path.length() > 4096 ||
    path[0] != '/' ||
    path[path.length() - 1] == '/' ||
    path.contains("//") ||
    path.contains("/./") ||
    path.contains("/../") ||
    path.has_suffix("/.") ||
    path.has_suffix("/..") {
    raise FusedCandidateExportFailed(OutputPath)
  }
  for unit in path {
    if !fused_export_path_unit_is_safe(unit) {
      raise FusedCandidateExportFailed(OutputPath)
    }
  }
  guard path.rev_find("/") is Some(separator) && separator < path.length() - 1 else {
    raise FusedCandidateExportFailed(OutputPath)
  }
  let parent = if separator == 0 { "/" } else { path[:separator].to_owned() }
  let basename = path[separator + 1:].to_owned()
  let actual_parent = @fs.realpath(parent) catch {
    _ => raise FusedCandidateExportFailed(OutputPath)
  }
  let output_exists = @fs.exists(path) catch {
    _ => raise FusedCandidateExportFailed(OutputPath)
  }
  if actual_parent != parent || output_exists {
    raise FusedCandidateExportFailed(OutputPath)
  }
  let staging = if parent == "/" {
    "/." + basename + ".lunaflux-fused-staging"
  } else {
    parent + "/." + basename + ".lunaflux-fused-staging"
  }
  let staging_exists = @fs.exists(staging) catch {
    _ => raise FusedCandidateExportFailed(OutputPath)
  }
  if staging_exists {
    raise FusedCandidateExportFailed(OutputPath)
  }
  staging
}
