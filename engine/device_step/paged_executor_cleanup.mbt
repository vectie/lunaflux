///|
fn remember_paged_failure(
  first : Ref[DeviceStepError?],
  error : DeviceStepError,
) -> Unit {
  if first.val is None {
    first.val = Some(error)
  }
}

///|
fn close_paged_functions(
  resources : PagedGraphResources,
  first : Ref[DeviceStepError?],
) -> Unit {
  for index = resources.functions.length() - 1; index >= 0; index = index - 1 {
    let function = resources.functions[index]
    if function.open {
      try function.resource.close() catch {
        error => remember_paged_failure(first, ExecutorDevice(Function, error))
      } noraise {
        _ => function.open = false
      }
    }
  }
}

///|
fn close_paged_modules(
  resources : PagedGraphResources,
  first : Ref[DeviceStepError?],
) -> Unit {
  for index = resources.modules.length() - 1; index >= 0; index = index - 1 {
    let module_owner = resources.modules[index]
    if module_owner.open &&
      !resources.functions.any(function => {
        function.open && function.module_index == index
      }) {
      try module_owner.resource.close() catch {
        error => remember_paged_failure(first, ExecutorDevice(Module, error))
      } noraise {
        _ => module_owner.open = false
      }
    }
  }
}

///|
fn close_paged_allocation(
  owner : PagedOwnedAllocation?,
  stage : PagedGraphResource,
  first : Ref[DeviceStepError?],
) -> Unit {
  match owner {
    Some(item) if item.open =>
      try item.resource.close() catch {
        error => remember_paged_failure(first, ExecutorDevice(stage, error))
      } noraise {
        _ => item.open = false
      }
    _ => ()
  }
}

///|
fn close_paged_descriptor(
  resources : PagedGraphResources,
  first : Ref[DeviceStepError?],
) -> Unit {
  match resources.descriptor {
    Some(Complete(owner)) =>
      try owner.close() catch {
        error => remember_paged_failure(first, error)
      } noraise {
        _ => resources.descriptor = None
      }
    Some(Failed(owner)) =>
      try owner.retry_close() catch {
        error => remember_paged_failure(first, error)
      } noraise {
        _ => resources.descriptor = None
      }
    None => ()
  }
}

///|
fn close_paged_lease(
  resources : PagedGraphResources,
  first : Ref[DeviceStepError?],
) -> Unit {
  match resources.weight_lease {
    Some(lease) if lease.open =>
      try lease.resource.close() catch {
        error =>
          remember_paged_failure(first, ExecutorDevice(WeightLease, error))
      } noraise {
        _ => lease.open = false
      }
    _ => ()
  }
}

///|
fn close_paged_stream(
  resources : PagedGraphResources,
  first : Ref[DeviceStepError?],
) -> Unit {
  match resources.stream {
    Some(stream) if stream.open =>
      try stream.resource.close() catch {
        error => remember_paged_failure(first, ExecutorDevice(Stream, error))
      } noraise {
        _ => stream.open = false
      }
    _ => ()
  }
}

///|
fn close_paged_resources(
  resources : PagedGraphResources,
) -> Unit raise DeviceStepError {
  let first : Ref[DeviceStepError?] = Ref(None)
  close_paged_functions(resources, first)
  close_paged_modules(resources, first)
  close_paged_allocation(resources.persistent_kv, PersistentKvArena, first)
  close_paged_allocation(
    resources.activation_workspace,
    ActivationWorkspaceArena,
    first,
  )
  close_paged_descriptor(resources, first)
  close_paged_lease(resources, first)
  close_paged_stream(resources, first)
  if first.val is Some(error) {
    raise error
  }
}

///|
pub fn PagedGraphExecutor::close(
  self : PagedGraphExecutor,
) -> Unit raise DeviceStepError {
  if self.lifecycle == GraphClosed {
    return
  }
  self.lifecycle = GraphClosingFailed
  close_paged_resources(self.resources)
  self.lifecycle = GraphClosed
}

///|
pub fn FailedPagedGraphPreparation::retry_close(
  self : FailedPagedGraphPreparation,
) -> Unit raise DeviceStepError {
  if self.lifecycle == GraphClosed {
    return
  }
  self.lifecycle = GraphClosingFailed
  close_paged_resources(self.resources) catch {
    error => {
      self.cleanup_failure = error
      raise error
    }
  }
  self.lifecycle = GraphClosed
}
