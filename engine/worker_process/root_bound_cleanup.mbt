///|
priv struct RootBoundCleanupAttempt {
  child_live : Bool
  roots_live : Bool
  child_failure : RootBoundWorkerProcessFailure?
  roots_failure : RootBoundWorkerProcessFailure?
}

///|
fn attempt_root_bound_cleanup(
  child_live : Bool,
  roots_live : Bool,
  close_child : () -> RootBoundWorkerProcessFailure?,
  close_roots : () -> RootBoundWorkerProcessFailure?,
  roots_is_live : () -> Bool,
) -> RootBoundCleanupAttempt {
  let mut next_child_live = child_live
  let mut child_failure = None
  let mut roots_failure = None
  if child_live {
    child_failure = close_child()
    if child_failure is None {
      next_child_live = false
    }
  }
  if roots_live {
    roots_failure = close_roots()
  }
  {
    child_live: next_child_live,
    roots_live: roots_is_live(),
    child_failure,
    roots_failure,
  }
}

///|
fn raise_cleanup_attempt(
  attempt : RootBoundCleanupAttempt,
) -> Unit raise RootBoundWorkerProcessError {
  if attempt.child_live {
    let child = attempt.child_failure.unwrap_or(
      Process(Transport(OperationFailed)),
    )
    if attempt.roots_failure is Some(roots) {
      raise RootBoundCompoundFailure(child, roots)
    }
    raise RootBoundFailure(child)
  }
  if attempt.roots_live {
    raise RootBoundFailure(attempt.roots_failure.unwrap_or(RootsBusy))
  }
  if attempt.roots_failure is Some(roots) {
    raise RootBoundFailure(roots)
  }
}
