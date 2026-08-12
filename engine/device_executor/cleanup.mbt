///|
fn remember_first(
  first : Ref[DeviceExecutorError?],
  error : DeviceExecutorError,
) -> Unit {
  if first.val is None {
    first.val = Some(error)
  }
}

///|
/// Generic reverse-order sibling closer used by production cleanup groups.
/// It attempts every independent sibling, preserves failed entries as open,
/// and reports only the first bounded failure after the pass.
fn[T] close_siblings_reverse(
  resources : Array[T],
  is_open : (T) -> Bool,
  close : (T) -> Unit raise DeviceExecutorError,
  mark_closed : (T) -> Unit,
  first : Ref[DeviceExecutorError?],
) -> Unit {
  for index = resources.length() - 1; index >= 0; index = index - 1 {
    let resource = resources[index]
    if is_open(resource) {
      try close(resource) catch {
        error => remember_first(first, error)
      } noraise {
        _ => mark_closed(resource)
      }
    }
  }
}

///|
fn[T] close_optional_resource(
  resource : T?,
  is_open : (T) -> Bool,
  close : (T) -> Unit raise DeviceExecutorError,
  mark_closed : (T) -> Unit,
  first : Ref[DeviceExecutorError?],
) -> Unit {
  match resource {
    Some(item) =>
      if is_open(item) {
        try close(item) catch {
          error => remember_first(first, error)
        } noraise {
          _ => mark_closed(item)
        }
      }
    None => ()
  }
}

///|
fn[Child, Parent] close_unblocked_parents_reverse(
  children : Array[Child],
  parents : Array[Parent],
  child_is_open : (Child) -> Bool,
  child_parent_index : (Child) -> Int,
  parent_is_open : (Parent) -> Bool,
  close : (Parent) -> Unit raise DeviceExecutorError,
  mark_closed : (Parent) -> Unit,
  first : Ref[DeviceExecutorError?],
) -> Unit {
  for index = parents.length() - 1; index >= 0; index = index - 1 {
    let parent = parents[index]
    if parent_is_open(parent) &&
      !children.any(child => {
        child_is_open(child) && child_parent_index(child) == index
      }) {
      try close(parent) catch {
        error => remember_first(first, error)
      } noraise {
        _ => mark_closed(parent)
      }
    }
  }
}

///|
fn close_functions(
  resources : ExecutorResources,
  first : Ref[DeviceExecutorError?],
) -> Unit {
  close_siblings_reverse(
    resources.functions,
    function => function.open,
    function => {
      function.resource.close() catch {
        error => raise CleanupFailed(Function, Device(error))
      }
    },
    function => function.open = false,
    first,
  )
}

///|
fn close_modules(
  resources : ExecutorResources,
  first : Ref[DeviceExecutorError?],
) -> Unit {
  close_unblocked_parents_reverse(
    resources.functions,
    resources.modules,
    function => function.open,
    function => function.module_index,
    item => item.open,
    item => {
      item.resource.close() catch {
        error => raise CleanupFailed(Module, Device(error))
      }
    },
    item => item.open = false,
    first,
  )
}

///|
fn close_vendor_plans(
  resources : ExecutorResources,
  first : Ref[DeviceExecutorError?],
) -> Unit {
  close_siblings_reverse(
    resources.vendor_plans,
    item => item.open,
    item => {
      item.resource.close() catch {
        error => raise CleanupFailed(VendorGemmPlan, Device(error))
      }
    },
    item => item.open = false,
    first,
  )
}

///|
fn close_vendor_handle(
  resources : ExecutorResources,
  first : Ref[DeviceExecutorError?],
) -> Unit {
  match resources.vendor_gemm {
    Some(handle) =>
      if handle.open && !resources.vendor_plans.any(plan => plan.open) {
        try handle.resource.close() catch {
          error =>
            remember_first(
              first,
              CleanupFailed(VendorGemmHandle, Device(error)),
            )
        } noraise {
          _ => handle.open = false
        }
      }
    None => ()
  }
}

///|
fn close_allocations_and_stream(
  resources : ExecutorResources,
  first : Ref[DeviceExecutorError?],
) -> Unit {
  close_optional_resource(
    resources.arena,
    arena => arena.open,
    arena => {
      arena.resource.close() catch {
        error => raise CleanupFailed(ActivationWorkspaceArena, Memory(error))
      }
    },
    arena => arena.open = false,
    first,
  )
  close_optional_resource(
    resources.token_staging,
    token_staging => token_staging.open,
    token_staging => {
      token_staging.resource.close() catch {
        error => raise CleanupFailed(TokenStaging, Device(error))
      }
    },
    token_staging => token_staging.open = false,
    first,
  )
  close_optional_resource(
    resources.stream,
    stream => stream.open,
    stream => {
      stream.resource.close() catch {
        error => raise CleanupFailed(Stream, Device(error))
      }
    },
    stream => stream.open = false,
    first,
  )
}

///|
fn close_all_retry_safe(
  resources : ExecutorResources,
) -> Unit raise DeviceExecutorError {
  let first : Ref[DeviceExecutorError?] = Ref(None)
  close_vendor_plans(resources, first)
  close_vendor_handle(resources, first)
  close_functions(resources, first)
  close_modules(resources, first)
  close_allocations_and_stream(resources, first)
  if first.val is Some(error) {
    raise error
  }
}

///|
fn close_with_lifecycle(
  lifecycle : ExecutorLifecycle,
  set_lifecycle : (ExecutorLifecycle) -> Unit,
  invalidate_output : () -> Unit,
  cleanup : () -> Unit raise DeviceExecutorError,
) -> Unit raise DeviceExecutorError {
  if lifecycle == Closed {
    return
  }
  invalidate_output()
  set_lifecycle(lifecycle_begin_close())
  cleanup()
  set_lifecycle(lifecycle_finish_close())
}

///|
pub fn PreparedDeviceExecutor::close(
  self : PreparedDeviceExecutor,
) -> Unit raise DeviceExecutorError {
  close_with_lifecycle(
    self.lifecycle,
    lifecycle => self.lifecycle = lifecycle,
    () => self.output_valid = false,
    () => close_all_retry_safe(self.resources),
  )
}

///|
pub fn FailedDevicePreparation::retry_close(
  self : FailedDevicePreparation,
) -> Unit raise DeviceExecutorError {
  close_with_lifecycle(
    self.lifecycle,
    lifecycle => self.lifecycle = lifecycle,
    () => (),
    () => {
      close_all_retry_safe(self.resources) catch {
        error => {
          self.cleanup_failure = error
          raise error
        }
      }
    },
  )
}
