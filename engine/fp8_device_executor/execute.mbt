///|
fn require_fp8_executor_ready(
  executor : Fp8DeviceExecutor,
) -> Unit raise Fp8DeviceExecutorError {
  if executor.lifecycle != ExecutorReady {
    raise InvalidFp8DeviceExecutor(ResourceState)
  }
}

///|
fn fp8_step(
  executor : Fp8DeviceExecutor,
  operation : @plan.OperationId,
) -> Fp8PreparedOperation raise Fp8DeviceExecutorError {
  for step in executor.steps {
    if step.operation == operation {
      return step
    }
  }
  raise operation_error(operation, Operation)
}

///|
fn upload_fp8_ephemeral(
  executor : Fp8DeviceExecutor,
  key : Fp8EphemeralStorageKey,
  source : FixedArray[Byte],
  source_offset : Int64,
) -> Unit raise Fp8DeviceExecutorError {
  require_fp8_executor_ready(executor)
  let allocation = fp8_ephemeral_allocation(executor.resources, key)
  if source_offset < 0L ||
    source_offset > source.length().to_int64() ||
    allocation.byte_count > source.length().to_int64() - source_offset {
    raise InvalidFp8DeviceExecutor(EphemeralRegion)
  }
  executor.resources.context.copy_from_fixed_host(
    allocation.resource,
    source,
    source_offset,
    0L,
    allocation.byte_count,
  ) catch {
    error => raise Fp8DeviceFailure(InputUpload, error)
  }
}

///|
pub fn Fp8DeviceExecutor::write_runtime_input(
  self : Fp8DeviceExecutor,
  input : @plan.OperationRuntimeInput,
  source : FixedArray[Byte],
  source_offset? : Int64 = 0L,
) -> Unit raise Fp8DeviceExecutorError {
  upload_fp8_ephemeral(self, Runtime(input), source, source_offset)
}

///|
pub fn Fp8DeviceExecutor::write_external_input(
  self : Fp8DeviceExecutor,
  input : @plan.ModelInputRef,
  source : FixedArray[Byte],
  source_offset? : Int64 = 0L,
) -> Unit raise Fp8DeviceExecutorError {
  upload_fp8_ephemeral(self, External(input), source, source_offset)
}

///|
pub fn Fp8DeviceExecutor::write_activation_input(
  self : Fp8DeviceExecutor,
  activation : @plan.ActivationRef,
  source : FixedArray[Byte],
  source_offset? : Int64 = 0L,
) -> Unit raise Fp8DeviceExecutorError {
  upload_fp8_ephemeral(self, Activation(activation), source, source_offset)
}

///|
fn poison_fp8_executor(executor : Fp8DeviceExecutor) -> Unit {
  executor.lifecycle = ExecutorPoisoned
}

///|
pub fn Fp8DeviceExecutor::execute(
  self : Fp8DeviceExecutor,
  operation : @plan.OperationId,
) -> ExecutedFp8Operation raise Fp8DeviceExecutorError {
  require_fp8_executor_ready(self)
  if self.epoch == 0xffffffffffffffffUL {
    poison_fp8_executor(self)
    raise InvalidFp8DeviceExecutor(Epoch)
  }
  let step = fp8_step(self, operation)
  let function = fp8_function(self.resources, step.function_index)
  let stream = fp8_stream(self.resources)
  self.lifecycle = ExecutorExecuting
  try
    function.launch_synchronous(step.dimensions, step.arguments, stream)
  catch {
    error => {
      poison_fp8_executor(self)
      raise Fp8DeviceFailure(Launch, error)
    }
  } noraise {
    _ => ()
  }
  let workspace = fp8_workspace_allocation(self.resources)
  try
    self.resources.context.copy_to_fixed_host(
      workspace,
      self.scale_scratch,
      step.workspace_offset,
      0L,
      step.workspace_bytes,
    )
  catch {
    error => {
      poison_fp8_executor(self)
      raise Fp8DeviceFailure(ScaleReadback, error)
    }
  } noraise {
    _ => ()
  }
  if !admit_fp8_execution_evidence(
      true,
      true,
      self.scale_scratch,
      step.workspace_bytes,
    ) {
    poison_fp8_executor(self)
    raise operation_error(operation, ScaleEvidence)
  }
  self.epoch += 1UL
  self.lifecycle = ExecutorExecuted
  { executor: self, operation, epoch: self.epoch }
}

///|
fn authenticate_executed_fp8(
  executor : Fp8DeviceExecutor,
  executed : ExecutedFp8Operation,
) -> Fp8PreparedOperation raise Fp8DeviceExecutorError {
  if !physical_equal(executed.executor, executor) {
    raise InvalidFp8DeviceExecutor(Owner)
  }
  if executor.lifecycle != ExecutorExecuted {
    raise InvalidFp8DeviceExecutor(ResourceState)
  }
  if executed.epoch != executor.epoch {
    raise InvalidFp8DeviceExecutor(Epoch)
  }
  fp8_step(executor, executed.operation)
}

///|
pub fn Fp8DeviceExecutor::copy_activation_output(
  self : Fp8DeviceExecutor,
  executed : ExecutedFp8Operation,
  output : @plan.ActivationRef,
  destination : FixedArray[Byte],
  destination_offset? : Int64 = 0L,
) -> Unit raise Fp8DeviceExecutorError {
  let step = authenticate_executed_fp8(self, executed)
  if !step.outputs.contains(output) {
    raise operation_error(step.operation, Output)
  }
  let allocation = fp8_ephemeral_allocation(self.resources, Activation(output))
  if destination_offset < 0L ||
    destination_offset > destination.length().to_int64() ||
    allocation.byte_count > destination.length().to_int64() - destination_offset {
    raise operation_error(step.operation, Output)
  }
  self.resources.context.copy_to_fixed_host(
    allocation.resource,
    destination,
    0L,
    destination_offset,
    allocation.byte_count,
  ) catch {
    error => {
      poison_fp8_executor(self)
      raise Fp8DeviceFailure(OutputReadback, error)
    }
  }
}

///|
pub fn Fp8DeviceExecutor::finish(
  self : Fp8DeviceExecutor,
  executed : ExecutedFp8Operation,
) -> Unit raise Fp8DeviceExecutorError {
  ignore(authenticate_executed_fp8(self, executed))
  self.lifecycle = ExecutorReady
}
