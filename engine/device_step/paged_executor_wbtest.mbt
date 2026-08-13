///|
priv struct FakePagedStep {
  operation : @plan.OperationId
}

///|
fn[Step] run_paged_step_range(
  steps : ReadOnlyArray[Step],
  lifecycle : PagedGraphLifecycle,
  set_lifecycle : (PagedGraphLifecycle) -> Unit,
  operation : (Step) -> @plan.OperationId,
  launch : (Step) -> Unit raise @device.DeviceError,
) -> Unit raise DeviceStepError {
  require_paged_lifecycle(lifecycle, GraphStaged)
  set_lifecycle(GraphExecuting)
  let mut index = 0
  while index < steps.length() {
    let step = steps[index]
    launch(step) catch {
      error => {
        set_lifecycle(GraphPoisoned)
        raise ExecutorLaunchFailed(operation(step), error)
      }
    }
    index += 1
  }
  set_lifecycle(GraphExecuted)
}

///|
fn dummy_paged_executor(lifecycle : PagedGraphLifecycle) -> PagedGraphExecutor {
  {
    resources: empty_paged_resources(),
    steps: ReadOnlyArray::from_array([]),
    lifecycle,
  }
}

///|
fn executor_test_submitted_plan() -> @worker_protocol.SubmittedSchedulePlan raise {
  let allocator = step_allocator()
  let buffer = @worker_protocol.SchedulePlanBuffer::new(step_protocol_limits())
  buffer.append_prefill(
    request_id=step_request(1UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[1],
    pages=[allocator.allocate_one()],
    capabilities=step_required_capability_array(),
    completion_slot=step_slot(0),
  )
  submit_step_buffer(buffer, token_budget=1)
}

///|
test "public paged capabilities reject lifecycle and foreign owners" {
  let closed = dummy_paged_executor(GraphClosed)
  try closed.stage(executor_test_submitted_plan()) catch {
    InvalidExecutorLifecycle(expected=GraphReady, actual=GraphClosed) => ()
    _ => fail("unexpected closed stage error")
  } noraise {
    _ => fail("closed executor admitted a stage")
  }
  let first = dummy_paged_executor(GraphStaged)
  let second = dummy_paged_executor(GraphStaged)
  let staged : StagedPagedGraphExecution = { executor: first, epoch: 1UL }
  try second.execute(staged) catch {
    ExecutorOwnerMismatch => ()
    _ => fail("unexpected foreign staged capability error")
  } noraise {
    _ => fail("foreign staged capability was admitted")
  }
  first.lifecycle = GraphExecuted
  second.lifecycle = GraphExecuted
  let executed : ExecutedPagedGraphExecution = { executor: first, epoch: 1UL }
  try second.finish(executed) catch {
    ExecutorOwnerMismatch => ()
    _ => fail("unexpected foreign executed capability error")
  } noraise {
    _ => fail("foreign executed capability was admitted")
  }
}

///|
test "paged execution launches semantic order and completes exactly once" {
  let lifecycle = Ref(GraphStaged)
  let launches : Array[Int] = []
  let steps = ReadOnlyArray::from_array([
    { operation: @plan.OperationId::new(0) },
    { operation: @plan.OperationId::new(1) },
    { operation: @plan.OperationId::new(2) },
  ])
  run_paged_step_range(
    steps,
    lifecycle.val,
    state => lifecycle.val = state,
    step => step.operation,
    step => launches.push(step.operation.as_int()),
  )
  assert_eq(launches, [0, 1, 2])
  assert_eq(lifecycle.val, GraphExecuted)
  try
    run_paged_step_range(
      steps,
      lifecycle.val,
      state => lifecycle.val = state,
      step => step.operation,
      _ => (),
    )
  catch {
    InvalidExecutorLifecycle(expected=GraphStaged, actual=GraphExecuted) => ()
    _ => fail("unexpected repeated execution error")
  } noraise {
    _ => fail("repeated execution was admitted")
  }
}

///|
test "paged launch failure poisons and reports exact operation" {
  let lifecycle = Ref(GraphStaged)
  let launches = Ref(0)
  let steps = ReadOnlyArray::from_array([
    { operation: @plan.OperationId::new(0) },
    { operation: @plan.OperationId::new(1) },
    { operation: @plan.OperationId::new(2) },
  ])
  try
    run_paged_step_range(
      steps,
      lifecycle.val,
      state => lifecycle.val = state,
      step => step.operation,
      step => {
        launches.val += 1
        if step.operation.as_int() == 1 {
          raise DriverFailure(AotKernelLaunch)
        }
      },
    )
  catch {
    ExecutorLaunchFailed(operation, DriverFailure(AotKernelLaunch)) =>
      assert_eq(operation.as_int(), 1)
    _ => fail("unexpected paged launch failure")
  } noraise {
    _ => fail("injected launch failure was swallowed")
  }
  assert_eq(launches.val, 2)
  assert_eq(lifecycle.val, GraphPoisoned)
}

///|
test "full blueprint builds one exact function map per entry point" {
  let metadata = blueprint_metadata()
  let plan = blueprint_model_plan(metadata)
  let layout = blueprint_layout(plan)
  let target = blueprint_target()
  let static_plan = full_static_plan(metadata, plan, layout, target)
  let memory = blueprint_memory_plan(metadata, plan, layout, target)
  let contracts = full_contract_set(plan, static_plan, memory, layout, target)
  let blueprint = admit_paged_graph_execution_blueprint(
    plan,
    blueprint_limits(),
    layout,
    static_plan,
    memory,
    blueprint_profile().id(),
    contracts,
    blueprint_artifacts(contracts),
  )
  let (functions, indices) = collect_paged_functions(blueprint)
  assert_eq(functions.length(), blueprint.steps().length())
  assert_eq(indices.length(), blueprint.steps().length())
  for index, function in functions {
    assert_eq(indices[index], index)
    assert_eq(function.identity, blueprint.steps()[index].entry_point())
    assert_eq(function.module_index, 0)
  }
}
