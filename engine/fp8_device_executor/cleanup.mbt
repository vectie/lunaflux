///|
fn remember_fp8_close_failure(
  first : Ref[Fp8DeviceExecutorError?],
  error : Fp8DeviceExecutorError,
) -> Unit {
  if first.val is None {
    first.val = Some(error)
  }
}

///|
fn close_fp8_functions(
  resources : Fp8DeviceResources,
  first : Ref[Fp8DeviceExecutorError?],
) -> Unit {
  for index = resources.functions.length() - 1; index >= 0; index = index - 1 {
    let function = resources.functions[index]
    if function.open {
      try function.resource.close() catch {
        error =>
          remember_fp8_close_failure(first, Fp8DeviceFailure(Function, error))
      } noraise {
        _ => function.open = false
      }
    }
  }
}

///|
fn close_fp8_modules(
  resources : Fp8DeviceResources,
  first : Ref[Fp8DeviceExecutorError?],
) -> Unit {
  for index = resources.modules.length() - 1; index >= 0; index = index - 1 {
    let module_ = resources.modules[index]
    if module_.open &&
      !resources.functions.any(function => {
        function.open && function.module_index == index
      }) {
      try module_.resource.close() catch {
        error =>
          remember_fp8_close_failure(first, Fp8DeviceFailure(Module, error))
      } noraise {
        _ => module_.open = false
      }
    }
  }
}

///|
fn close_fp8_owned_allocation(
  owner : Fp8OwnedAllocation?,
  stage : Fp8DeviceExecutorResource,
  first : Ref[Fp8DeviceExecutorError?],
) -> Unit {
  match owner {
    Some(allocation) if allocation.open =>
      try allocation.resource.close() catch {
        error =>
          remember_fp8_close_failure(first, Fp8DeviceFailure(stage, error))
      } noraise {
        _ => allocation.open = false
      }
    _ => ()
  }
}

///|
fn close_fp8_ephemeral(
  resources : Fp8DeviceResources,
  first : Ref[Fp8DeviceExecutorError?],
) -> Unit {
  for index = resources.ephemeral.length() - 1; index >= 0; index = index - 1 {
    close_fp8_owned_allocation(
      Some(resources.ephemeral[index]),
      Ephemeral,
      first,
    )
  }
}

///|
fn close_fp8_stream(
  resources : Fp8DeviceResources,
  first : Ref[Fp8DeviceExecutorError?],
) -> Unit {
  match resources.stream {
    Some(stream) if stream.open =>
      try stream.resource.close() catch {
        error =>
          remember_fp8_close_failure(first, Fp8DeviceFailure(Stream, error))
      } noraise {
        _ => stream.open = false
      }
    _ => ()
  }
}

///|
fn close_fp8_weight_owner(
  resources : Fp8DeviceResources,
  first : Ref[Fp8DeviceExecutorError?],
) -> Unit {
  match resources.weight_lease {
    Some(lease) if lease.open =>
      try lease.resource.close() catch {
        error =>
          remember_fp8_close_failure(
            first,
            Fp8DeviceFailure(WeightLease, error),
          )
      } noraise {
        _ => lease.open = false
      }
    _ => ()
  }
  let lease_open = match resources.weight_lease {
    Some(lease) => lease.open
    None => false
  }
  if !lease_open {
    match resources.weights {
      Some(weights) =>
        try weights.close() catch {
          error => remember_fp8_close_failure(first, Fp8WeightFailure(error))
        } noraise {
          _ => resources.weights = None
        }
      None => ()
    }
  }
}

///|
fn close_fp8_resources(
  resources : Fp8DeviceResources,
) -> Unit raise Fp8DeviceExecutorError {
  let first : Ref[Fp8DeviceExecutorError?] = Ref(None)
  close_fp8_functions(resources, first)
  close_fp8_modules(resources, first)
  close_fp8_ephemeral(resources, first)
  close_fp8_owned_allocation(resources.workspace, Workspace, first)
  close_fp8_stream(resources, first)
  close_fp8_weight_owner(resources, first)
  if first.val is Some(error) {
    raise error
  }
}

///|
fn fp8_resources_closed(resources : Fp8DeviceResources) -> Bool {
  let stream_closed = match resources.stream {
    Some(stream) => !stream.open
    None => true
  }
  let lease_closed = match resources.weight_lease {
    Some(lease) => !lease.open
    None => true
  }
  let workspace_closed = match resources.workspace {
    Some(workspace) => !workspace.open
    None => true
  }
  resources.weights is None &&
  stream_closed &&
  lease_closed &&
  workspace_closed &&
  resources.ephemeral.all(allocation => !allocation.open) &&
  resources.modules.all(module_ => !module_.open) &&
  resources.functions.all(function => !function.open)
}

///|
fn fp8_construction_failed(
  resources : Fp8DeviceResources,
  cause : Fp8DeviceExecutorError,
) -> Fp8DeviceExecutorPreparation raise Fp8DeviceExecutorError {
  try close_fp8_resources(resources) catch {
    cleanup_failure =>
      Fp8ExecutorCleanupRequired({
        resources,
        cause,
        cleanup_failure,
        closed: fp8_resources_closed(resources),
      })
  } noraise {
    _ => raise cause
  }
}

///|
pub fn Fp8DeviceExecutor::close(
  self : Fp8DeviceExecutor,
) -> Unit raise Fp8DeviceExecutorError {
  if self.lifecycle == ExecutorClosed {
    return
  }
  if self.lifecycle == ExecutorExecuting {
    raise InvalidFp8DeviceExecutor(ResourceState)
  }
  self.lifecycle = ExecutorClosingFailed
  close_fp8_resources(self.resources)
  self.lifecycle = ExecutorClosed
}

///|
fn remember_fp8_cleanup_failure(
  failed : FailedFp8DeviceExecutorPreparation,
  error : Fp8DeviceExecutorError,
) -> Unit raise Fp8DeviceExecutorError {
  failed.cleanup_failure = error
  raise error
}

///|
pub fn FailedFp8DeviceExecutorPreparation::retry_close(
  self : FailedFp8DeviceExecutorPreparation,
) -> Unit raise Fp8DeviceExecutorError {
  if self.closed {
    return
  }
  try close_fp8_resources(self.resources) catch {
    error => remember_fp8_cleanup_failure(self, error)
  } noraise {
    _ => self.closed = true
  }
}
