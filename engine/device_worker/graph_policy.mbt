///|
/// Selects only modes representable by the current authenticated production
/// manifest. Every admitted Luna authorization already carries the exact
/// validated eager-fallback disposition; capture safety decides whether that
/// fallback surrounds a startup capture attempt or remains the eager primary.
fn select_authenticated_bf16_graph_policy(
  has_luna_authorization : Bool,
  capture_safe : Bool,
) -> @device_step.PagedGraphExecutionPolicy {
  if !has_luna_authorization || !capture_safe {
    PagedOrderedEagerOnly
  } else {
    PagedCapturedWithEagerFallback
  }
}

///|
fn authenticated_bf16_graph_policy(
  execution : @execution_manifest_file.PagedExecutionAdmission,
) -> @device_step.PagedGraphExecutionPolicy {
  match execution.luna_authorization() {
    None => select_authenticated_bf16_graph_policy(false, false)
    Some(authorization) =>
      // Artifact admission currently has exactly one execution disposition.
      // Keep the check at this consumer boundary so a future disposition
      // cannot silently acquire eager-fallback authority here.
      match authorization.execution_disposition() {
        LunaValidatedEagerFallback =>
          select_authenticated_bf16_graph_policy(
            true,
            authorization.graph_metadata().capture_safe(),
          )
      }
  }
}
