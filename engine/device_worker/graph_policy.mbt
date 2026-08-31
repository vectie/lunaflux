///|
/// Selects only modes representable by the current authenticated production
/// manifest. Every admitted Luna authorization already carries the exact
/// validated eager-fallback disposition; capture safety decides whether that
/// fallback surrounds a startup capture attempt or remains the eager primary.
fn select_authenticated_bf16_graph_policy(
  has_luna_authorization : Bool,
  capture_safe : Bool,
) -> @device_step.PagedGraphExecutionPolicy {
  // Capture is legal only when the authenticated execution authorization also
  // carries the bounded graph-memory metadata consumed by startup capacity
  // admission. An ordinary AOT launch list remains eager even when its
  // pointers are stable; stability alone is not a memory-accounting contract.
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
    // Without authenticated graph metadata there is no bounded capture-memory
    // contract. Keep the ordinary AOT launch list ordered and eager.
    None => select_authenticated_bf16_graph_policy(false, true)
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
