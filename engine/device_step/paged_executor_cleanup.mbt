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
fn run_reverse_paged_executor_close(
  count : Int,
  close_one : (Int) -> Unit,
) -> Unit {
  for index = count - 1; index >= 0; index = index - 1 {
    close_one(index)
  }
}

///|
fn close_paged_ordered_executors(
  resources : PagedGraphResources,
  first : Ref[DeviceStepError?],
) -> Unit {
  run_reverse_paged_executor_close(resources.ordered_executors.length(), index => {
    let owner = resources.ordered_executors[index]
    if owner.open {
      let mut drained = false
      try owner.resource.abort() catch {
        error =>
          remember_paged_failure(first, ExecutorDevice(OrderedExecutor, error))
      } noraise {
        _ => drained = true
      }
      if drained {
        try owner.resource.close() catch {
          error =>
            remember_paged_failure(
              first,
              ExecutorDevice(OrderedExecutor, error),
            )
        } noraise {
          _ => owner.open = false
        }
      }
    }
  })
}

///|
fn paged_ordered_executor_open(resources : PagedGraphResources) -> Bool {
  resources.ordered_executors.any(owner => owner.open)
}

///|
/// Preserves every launch dependency until the ordered executor has drained
/// and released its native leases. Close failures therefore retain one exact
/// retryable aggregate rather than cascading Busy errors across dependencies.
fn run_paged_ordered_dependency_close(
  close_ordered : () -> Unit,
  ordered_open : () -> Bool,
  close_dependencies : () -> Unit,
) -> Unit {
  close_ordered()
  if !ordered_open() {
    close_dependencies()
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
fn paged_weight_lease_open(resources : PagedGraphResources) -> Bool {
  match resources.weight_lease {
    Some(lease) => lease.open
    None => false
  }
}

///|
/// Preserves the lifetime dependency between the allocation lease and its
/// numeric-weight owner. `close_lease` records rather than raises its failure,
/// so the post-attempt liveness observation is the only authority to proceed.
fn run_paged_weight_dependency_close(
  close_lease : () -> Unit,
  lease_open : () -> Bool,
  close_numeric : () -> Unit,
) -> Unit {
  close_lease()
  if !lease_open() {
    close_numeric()
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
fn close_numeric_weights(
  resources : PagedGraphResources,
  first : Ref[DeviceStepError?],
) -> Unit {
  match resources.numeric_weights {
    Some(I8NumericWeights(owner)) =>
      try owner.close() catch {
        error => remember_paged_failure(first, error)
      } noraise {
        _ => resources.numeric_weights = None
      }
    Some(Bf16NumericWeights(owner)) =>
      try owner.close() catch {
        _ => remember_paged_failure(first, NumericWeightClose)
      } noraise {
        _ => resources.numeric_weights = None
      }
    Some(Fp8NumericWeights(owner)) =>
      try owner.close() catch {
        _ => remember_paged_failure(first, NumericWeightClose)
      } noraise {
        _ => resources.numeric_weights = None
      }
    None => ()
  }
}

///|
/// Applies the dependency-ordered close sequence shared by production cleanup
/// and the host failure-injection tests. Individual close stages retain their
/// own retryable state and accumulate the first error without short-circuiting.
fn run_paged_close_order(stages : ReadOnlyArray[() -> Unit]) -> Unit {
  for stage in stages {
    stage()
  }
}

///|
fn close_paged_resources(
  resources : PagedGraphResources,
) -> Unit raise DeviceStepError {
  let first : Ref[DeviceStepError?] = Ref(None)
  run_paged_ordered_dependency_close(
    () => close_paged_ordered_executors(resources, first),
    () => paged_ordered_executor_open(resources),
    () => {
      run_paged_close_order(
        ReadOnlyArray::from_array([
          () => close_paged_functions(resources, first),
          () => close_paged_modules(resources, first),
          () => {
            close_paged_allocation(
              resources.greedy_sampling_results,
              SamplingResult,
              first,
            )
          },
          () => {
            close_paged_allocation(
              resources.fused_dispatch_canary,
              FusedDispatchCanary,
              first,
            )
          },
          () => {
            close_paged_allocation(
              resources.persistent_kv,
              PersistentKvArena,
              first,
            )
          },
          () => {
            close_paged_allocation(
              resources.activation_workspace,
              ActivationWorkspaceArena,
              first,
            )
          },
          () => close_paged_descriptor(resources, first),
          () => {
            run_paged_weight_dependency_close(
              () => close_paged_lease(resources, first),
              () => paged_weight_lease_open(resources),
              () => close_numeric_weights(resources, first),
            )
          },
          () => close_paged_stream(resources, first),
        ]),
      )
    },
  )
  if paged_ordered_executor_open(resources) {
    guard first.val is Some(error) else { raise InvalidExecutor(ResourceState) }
    raise error
  }
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
  let outcome : Result[Unit, DeviceStepError] = Ok(
    close_paged_resources(self.resources),
  ) catch {
    error => Err(error)
  }
  match outcome {
    Err(error) => {
      self.cleanup_failure = error
      raise error
    }
    Ok(_) => ()
  }
  self.lifecycle = GraphClosed
}
