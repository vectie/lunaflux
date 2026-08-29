///|
fn run_cleanup_chain(
  collective : () -> Unit raise TensorParallelDeviceWorkerError,
  executor : () -> Unit raise TensorParallelDeviceWorkerError,
  functions : () -> Unit raise TensorParallelDeviceWorkerError,
  modules : () -> Unit raise TensorParallelDeviceWorkerError,
  stream : () -> Unit raise TensorParallelDeviceWorkerError,
  kv : () -> Unit raise TensorParallelDeviceWorkerError,
  activation : () -> Unit raise TensorParallelDeviceWorkerError,
  constants : () -> Unit raise TensorParallelDeviceWorkerError,
  descriptor : () -> Unit raise TensorParallelDeviceWorkerError,
  weights : () -> Unit raise TensorParallelDeviceWorkerError,
  context : () -> Unit raise TensorParallelDeviceWorkerError,
) -> Unit raise TensorParallelDeviceWorkerError {
  collective()
  executor()
  functions()
  modules()
  stream()
  kv()
  activation()
  constants()
  descriptor()
  weights()
  context()
}

///|
fn close_executor_owner(
  owner : TensorParallelOwnedExecutor,
) -> Unit raise TensorParallelDeviceWorkerError {
  if !owner.open {
    return
  }
  if owner.abort_required {
    owner.resource.abort() catch {
      error => raise Device(KernelExecutor, error)
    }
    owner.abort_required = false
  }
  owner.resource.close() catch {
    error => raise Device(KernelExecutor, error)
  }
  owner.open = false
}

///|
fn close_collective_owner(
  owner : TensorParallelOwnedCollective,
) -> Unit raise TensorParallelDeviceWorkerError {
  if !owner.open {
    return
  }
  let mut failure : TensorParallelDeviceWorkerError? = None
  try {
    if owner.abort_required {
      owner.resource.abort()
    } else {
      owner.resource.close()
    }
  } catch {
    error => failure = Some(Device(Collective, error))
  } noraise {
    _ => ()
  }
  // Both native terminal methods consume the communicator even when destroy
  // reports failure. Retrying the communicator itself would be invalid.
  owner.open = false
  match failure {
    Some(error) => raise error
    None => ()
  }
}

///|
fn close_functions(
  functions : Array[TensorParallelOwnedFunction],
) -> Unit raise TensorParallelDeviceWorkerError {
  for index = functions.length() - 1; index >= 0; index = index - 1 {
    let function = functions[index]
    if function.open {
      function.resource.close() catch {
        error => raise Device(Function, error)
      }
      function.open = false
    }
  }
}

///|
fn close_modules(
  modules : Array[TensorParallelOwnedModule],
  functions : Array[TensorParallelOwnedFunction],
) -> Unit raise TensorParallelDeviceWorkerError {
  for index = modules.length() - 1; index >= 0; index = index - 1 {
    let loaded_module = modules[index]
    if loaded_module.open {
      if functions.any(function => {
          function.open && function.module_index == index
        }) {
        raise Invalid(ResourceState)
      }
      loaded_module.resource.close() catch {
        error => raise Device(Module, error)
      }
      loaded_module.open = false
    }
  }
}

///|
fn close_stream_owner(
  owner : TensorParallelOwnedStream,
) -> Unit raise TensorParallelDeviceWorkerError {
  if owner.open {
    owner.resource.close() catch {
      error => raise Device(Stream, error)
    }
    owner.open = false
  }
}

///|
fn close_allocation_owner(
  owner : TensorParallelOwnedAllocation,
  stage : TensorParallelDeviceWorkerResource,
) -> Unit raise TensorParallelDeviceWorkerError {
  if owner.open {
    owner.resource.close() catch {
      error => raise Device(stage, error)
    }
    owner.open = false
  }
}

///|
fn close_partial_collective(
  resources : TensorParallelPartialResources,
) -> Unit raise TensorParallelDeviceWorkerError {
  guard resources.collective is Some(owner) else { return }
  defer {
    resources.collective = None
  }
  close_collective_owner(owner)
}

///|
fn close_partial_executor(
  resources : TensorParallelPartialResources,
) -> Unit raise TensorParallelDeviceWorkerError {
  guard resources.executor is Some(owner) else { return }
  close_executor_owner(owner)
  resources.executor = None
}

///|
fn close_partial_stream(
  resources : TensorParallelPartialResources,
) -> Unit raise TensorParallelDeviceWorkerError {
  guard resources.stream is Some(owner) else { return }
  close_stream_owner(owner)
  resources.stream = None
}

///|
fn close_partial_kv(
  resources : TensorParallelPartialResources,
) -> Unit raise TensorParallelDeviceWorkerError {
  match resources.kv {
    Some(KvComplete(owner)) => owner.close() catch { error => raise Kv(error) }
    Some(KvFailed(owner)) =>
      owner.retry_close() catch {
        error => raise Kv(error)
      }
    None => return
  }
  resources.kv = None
}

///|
fn close_partial_allocation(
  resources : TensorParallelPartialResources,
  activation : Bool,
) -> Unit raise TensorParallelDeviceWorkerError {
  let selected = if activation {
    resources.activation_workspace
  } else {
    resources.rank_constants
  }
  guard selected is Some(owner) else { return }
  let stage = if activation {
    TensorParallelDeviceWorkerResource::ActivationWorkspace
  } else {
    RankConstants
  }
  close_allocation_owner(owner, stage)
  if activation {
    resources.activation_workspace = None
  } else {
    resources.rank_constants = None
  }
}

///|
fn close_partial_descriptor(
  resources : TensorParallelPartialResources,
) -> Unit raise TensorParallelDeviceWorkerError {
  match resources.descriptor {
    Some(DescriptorComplete(owner)) =>
      owner.close() catch {
        error => raise Descriptor(error)
      }
    Some(DescriptorFailed(owner)) =>
      owner.retry_close() catch {
        error => raise Descriptor(error)
      }
    None => return
  }
  resources.descriptor = None
}

///|
fn close_partial_weights(
  resources : TensorParallelPartialResources,
) -> Unit raise TensorParallelDeviceWorkerError {
  match resources.weights {
    Some(WeightComplete(owner)) =>
      owner.close() catch {
        error => raise Weights(error)
      }
    Some(WeightFailed(owner)) =>
      owner.retry_close() catch {
        error => raise Weights(error)
      }
    None => return
  }
  resources.weights = None
}

///|
fn close_partial_context(
  resources : TensorParallelPartialResources,
) -> Unit raise TensorParallelDeviceWorkerError {
  guard resources.context is Some(context) else { return }
  context.close() catch {
    error => raise Device(Context, error)
  }
  resources.context = None
}

///|
fn close_partial_resources(
  resources : TensorParallelPartialResources,
) -> Unit raise TensorParallelDeviceWorkerError {
  run_cleanup_chain(
    () => close_partial_collective(resources),
    () => close_partial_executor(resources),
    () => close_functions(resources.functions),
    () => close_modules(resources.modules, resources.functions),
    () => close_partial_stream(resources),
    () => close_partial_kv(resources),
    () => close_partial_allocation(resources, true),
    () => close_partial_allocation(resources, false),
    () => close_partial_descriptor(resources),
    () => close_partial_weights(resources),
    () => close_partial_context(resources),
  )
}

///|
fn close_live_resources(
  resources : TensorParallelLiveResources,
) -> Unit raise TensorParallelDeviceWorkerError {
  run_cleanup_chain(
    () => close_collective_owner(resources.collective),
    () => close_executor_owner(resources.executor),
    () => close_functions(resources.functions),
    () => close_modules(resources.modules, resources.functions),
    () => close_stream_owner(resources.stream),
    () => resources.kv.close() catch { error => raise Kv(error) },
    () => {
      close_allocation_owner(
        resources.activation_workspace,
        ActivationWorkspace,
      )
    },
    () => close_allocation_owner(resources.rank_constants, RankConstants),
    () => {
      resources.descriptor.close() catch {
        error => raise Descriptor(error)
      }
    },
    () => resources.weights.close() catch { error => raise Weights(error) },
    () => {
      resources.context.close() catch {
        error => raise Device(Context, error)
      }
    },
  )
}

///|
fn resources_complete(resources : TensorParallelLiveResources) -> Bool {
  resources.collective.open &&
  resources.executor.open &&
  resources.stream.open &&
  resources.activation_workspace.open &&
  resources.rank_constants.open &&
  resources.kv.is_usable() &&
  resources.functions.all(function => function.open) &&
  resources.modules.all(loaded_module => loaded_module.open)
}

///|
fn preparation_failed(
  resources : TensorParallelPartialResources,
  cause : TensorParallelDeviceWorkerError,
) -> TensorParallelDeviceWorkerPreparation raise TensorParallelDeviceWorkerError {
  try close_partial_resources(resources) catch {
    cleanup_failure =>
      TensorParallelWorkerCleanupRequired({
        resources,
        cause,
        cleanup_failure,
        lifecycle: WorkerClosingFailed,
      })
  } noraise {
    _ => raise cause
  }
}

///|
pub fn FailedTensorParallelDeviceWorkerPreparation::retry_close(
  self : FailedTensorParallelDeviceWorkerPreparation,
) -> Unit raise TensorParallelDeviceWorkerError {
  if self.lifecycle == WorkerClosed {
    return
  }
  self.lifecycle = WorkerClosingFailed
  close_partial_resources(self.resources)
  self.lifecycle = WorkerClosed
}

///|
pub fn TensorParallelDeviceWorkerOwner::abort_plan(
  self : TensorParallelDeviceWorkerOwner,
) -> Unit raise TensorParallelDeviceWorkerError {
  match self.lifecycle {
    WorkerStaged
    | WorkerExecuting
    | WorkerCollectivePending
    | WorkerExecuted
    | WorkerDevicePending
    | WorkerFaulted => ()
    WorkerDraining => return
    _ => raise Invalid(ResourceState)
  }
  self.resources.collective.abort_required = true
  self.resources.executor.abort_required = true
  self.resources.collective.resource.invalidate() catch {
    error => {
      let translated = Device(Collective, error)
      if self.cleanup_failure is None {
        self.cleanup_failure = Some(translated)
      }
      raise translated
    }
  }
  self.lifecycle = WorkerDraining
}

///|
pub fn TensorParallelDeviceWorkerOwner::begin_drain(
  self : TensorParallelDeviceWorkerOwner,
) -> Unit raise TensorParallelDeviceWorkerError {
  match self.lifecycle {
    WorkerReady | WorkerFaulted => self.lifecycle = WorkerDraining
    WorkerDraining => ()
    _ => raise Invalid(ResourceState)
  }
}

///|
pub fn TensorParallelDeviceWorkerOwner::close(
  self : TensorParallelDeviceWorkerOwner,
) -> Unit raise TensorParallelDeviceWorkerError {
  if self.lifecycle == WorkerClosed {
    return
  }
  match self.lifecycle {
    WorkerStarting
    | WorkerStaged
    | WorkerExecuting
    | WorkerCollectivePending
    | WorkerDevicePending
    | WorkerExecuted
    | WorkerFaulted => {
      self.resources.collective.abort_required = true
      self.resources.executor.abort_required = true
    }
    _ => ()
  }
  self.lifecycle = WorkerClosingFailed
  let outcome : Result[Unit, TensorParallelDeviceWorkerError] = Ok(
    close_live_resources(self.resources),
  ) catch {
    error => Err(error)
  }
  match outcome {
    Err(error) => {
      if self.cleanup_failure is None {
        self.cleanup_failure = Some(error)
      }
      raise error
    }
    Ok(_) => ()
  }
  self.lifecycle = WorkerClosed
}
