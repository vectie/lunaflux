///|
/// Selects the backend execution mode for one immutable AOT launch list.
///
/// External Luna authorization authenticates an optional specialization; it
/// is not permission to use a CUDA execution primitive. Stable function,
/// geometry, argument, allocation, and stream ownership are already enforced
/// by the private ordered executor. Consequently an ordinary AOT graph may be
/// captured at startup with an eager fallback even when no specialization was
/// supplied. An authorization that explicitly marks its specialization unsafe
/// for capture still forces eager execution.
fn select_authenticated_bf16_graph_policy(
  _has_luna_authorization : Bool,
  capture_safe : Bool,
) -> @device_step.PagedGraphExecutionPolicy {
  if !capture_safe {
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
    // The baseline AOT graph owns only startup-stable backend resources. It may
    // use capture without claiming specialization shape or graph-memory
    // metadata; telemetry therefore retains no asserted specialization shape.
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
