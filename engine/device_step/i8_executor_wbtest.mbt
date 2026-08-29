///|
priv struct FakeI8ConstructionState {
  open : Array[Bool]
  events : Array[Int]
  mut cleanup_failure : Int
}

///|
fn close_fake_i8_construction(
  state : FakeI8ConstructionState,
  fail_close? : Int = 0,
) -> Unit {
  let stages : Array[() -> Unit] = []
  for index in 0..<state.open.length() {
    let resource = state.open.length() - index
    stages.push(() => {
      if state.open[resource - 1] {
        state.events.push(-resource)
        if resource == fail_close && state.cleanup_failure == 0 {
          state.cleanup_failure = resource
        } else {
          state.open[resource - 1] = false
        }
      }
    })
  }
  run_paged_close_order(ReadOnlyArray::from_array(stages))
}

///|
fn injected_i8_construction(
  fail_stage : Int,
  fail_close? : Int = 0,
) -> FakeI8ConstructionState raise {
  let state : FakeI8ConstructionState = {
    open: Array::make(9, false),
    events: [],
    cleanup_failure: 0,
  }
  let stages : Array[() -> Unit raise DeviceStepError] = []
  for stage in 1..<=9 {
    stages.push(() => {
      state.events.push(stage)
      if stage == fail_stage {
        raise InvalidExecutor(ResourceState)
      }
      state.open[stage - 1] = true
    })
  }
  ignore(
    run_i8_construction_stages(ReadOnlyArray::from_array(stages), error => {
      if !(error is InvalidExecutor(ResourceState)) {
        state.cleanup_failure = -1
      }
      close_fake_i8_construction(state, fail_close~)
      state
    }),
  )
  state
}

///|
test "I8 native fake seam construction cleans every real boundary in reverse" {
  for failed in 1..<=9 {
    let state = injected_i8_construction(failed)
    assert_eq(state.events.length(), failed + failed - 1)
    for opened in 1..<failed {
      assert_eq(state.events[opened - 1], opened)
      assert_eq(state.events[state.events.length() - opened], -opened)
    }
    assert_false(state.open.any(value => value))
  }
}

///|
test "I8 construction cleanup failure is retained and retryable" {
  let state = injected_i8_construction(9, fail_close=4)
  assert_eq(state.cleanup_failure, 4)
  assert_true(state.open[3])
  assert_eq(
    state.open.fold(init=0, (count, value) => {
      if value {
        count + 1
      } else {
        count
      }
    }),
    1,
  )
  close_fake_i8_construction(state)
  assert_false(state.open.any(value => value))
  assert_eq(state.events[state.events.length() - 1], -4)
}

///|
test "I8 admitted steps bind exact scale region immediately after each weight" {
  let fixture = i8_blueprint_fixture()
  let blueprint = fixture.blueprint
  let weight_layout = blueprint.weight_layout()
  let mut scale_count = 0
  for step in blueprint.steps() {
    let operands = step.operands()
    for index, operand in operands {
      if operand.role() is WeightScaleInput(weight, scale) {
        assert_true(index > 0)
        assert_eq(operands[index - 1].role(), WeightInput(weight))
        guard weight_layout.region(scale.tensor()) is Some(region) else {
          fail("missing admitted scale region")
        }
        assert_eq(operand.space(), WeightArena)
        assert_eq(operand.offset_bytes(), region.offset_bytes())
        assert_eq(operand.byte_count(), region.byte_length())
        scale_count += 1
      }
    }
  }
  assert_eq(scale_count, fixture.policy.scale_tensor_count())
}

///|
test "I8 blueprint retains exact memory and artifact authorities" {
  let fixture = i8_blueprint_fixture()
  assert_eq(fixture.blueprint.memory_plan, fixture.memory_plan)
  assert_true(physical_equal(fixture.blueprint.artifacts, fixture.artifacts))
  let alternate_memory = i8_fixture_memory(
    fixture.static_plan,
    maximum=fixture.memory_plan.limits().max_arena_bytes() * 2L,
  )
  assert_false(alternate_memory == fixture.blueprint.memory_plan)
  let alternate_artifacts = i8_fixture_artifacts(
    fixture.contracts,
    alternate_symbols=true,
  )
  assert_false(physical_equal(alternate_artifacts, fixture.blueprint.artifacts))
}

///|
test "production I8 blueprint rejects any unselected launch profile" {
  let fixture = i8_blueprint_fixture()
  let second = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(2),
    max_query_rows=fixture.profile.max_query_rows(),
    max_query_tokens=fixture.profile.max_query_tokens(),
    max_page_table_entries=fixture.profile.max_page_table_entries(),
  )
  let resolved = i8_fixture_resolved(
    fixture.plan,
    fixture.layout,
    fixture.contracts.device_target(),
  )
  let contracts = i8_fixture_contracts(
    fixture.plan,
    resolved,
    fixture.static_plan,
    fixture.memory_plan,
    fixture.layout,
    [fixture.profile, second],
  )
  let artifacts = i8_fixture_artifacts(contracts)
  try
    admit_i8_paged_graph_execution_blueprint_v1(
      fixture.plan,
      fixture.device_numeric,
      fixture.policy,
      blueprint_limits(),
      fixture.layout,
      fixture.static_plan,
      fixture.memory_plan,
      fixture.profile.id(),
      contracts,
      artifacts,
    )
  catch {
    InvalidBlueprint(Profile) => ()
    _ => fail("unexpected multi-profile I8 rejection")
  } noraise {
    _ => fail("unselected production-I8 launch profile was retained")
  }
}

///|
test "I8 host launch canary uses admitted semantic order including scale steps" {
  let steps = i8_blueprint_fixture().blueprint.steps()
  let lifecycle = Ref(GraphStaged)
  let launches : Array[Int] = []
  run_paged_ordered_step_range(
    steps,
    lifecycle.val,
    state => lifecycle.val = state,
    (_, step) => launches.push(step.operation_id().as_int()),
    () => (),
    () => (),
    () => (),
  )
  assert_eq(launches.length(), steps.length())
  for index, operation in launches {
    assert_eq(operation, index)
  }
  assert_true(
    steps.any(step => {
      step.operands().any(operand => operand.role() is WeightScaleInput(_, _))
    }),
  )
  assert_eq(lifecycle.val, GraphExecuted)
}

///|
test "I8 host launch failure poisons before any later admitted operation" {
  let steps = i8_blueprint_fixture().blueprint.steps()
  let lifecycle = Ref(GraphStaged)
  let launches : Array[Int] = []
  let failed_operation = steps[2].operation_id()
  try
    run_paged_ordered_step_range(
      steps,
      lifecycle.val,
      state => lifecycle.val = state,
      (_, step) => {
        launches.push(step.operation_id().as_int())
        if step.operation_id() == failed_operation {
          raise ExecutorLaunchFailed(
            failed_operation,
            DriverFailure(AotKernelLaunch),
          )
        }
      },
      () => (),
      () => (),
      () => (),
    )
  catch {
    ExecutorLaunchFailed(operation, DriverFailure(AotKernelLaunch)) =>
      assert_eq(operation, failed_operation)
    _ => fail("unexpected I8 launch-canary failure")
  } noraise {
    _ => fail("forced I8 launch failure was swallowed")
  }
  assert_eq(launches, [0, 1, 2])
  assert_eq(lifecycle.val, GraphPoisoned)
}

///|
test "empty I8 construction cleanup is deterministic and idempotent" {
  let resources = empty_paged_resources()
  close_paged_resources(resources)
  close_paged_resources(resources)
  assert_true(resources.numeric_weights is None)
}

///|
test "ordered executor close retains every dependency until leases release" {
  let ordered_open = Ref(true)
  let close_attempts = Ref(0)
  let dependency_closes = Ref(0)
  run_paged_ordered_dependency_close(
    () => close_attempts.val += 1,
    () => ordered_open.val,
    () => dependency_closes.val += 1,
  )
  assert_eq(close_attempts.val, 1)
  assert_eq(dependency_closes.val, 0)
  ordered_open.val = false
  run_paged_ordered_dependency_close(
    () => close_attempts.val += 1,
    () => ordered_open.val,
    () => dependency_closes.val += 1,
  )
  assert_eq(close_attempts.val, 2)
  assert_eq(dependency_closes.val, 1)
}

///|
test "live weight lease gates numeric owner close until retry succeeds" {
  let lease_open = Ref(true)
  let fail_lease_close = Ref(true)
  let numeric_open = Ref(true)
  let events : Array[String] = []
  let close_attempt = () => {
    run_paged_weight_dependency_close(
      () => {
        events.push("lease")
        if !fail_lease_close.val {
          lease_open.val = false
        }
      },
      () => lease_open.val,
      () => {
        events.push("numeric")
        numeric_open.val = false
      },
    )
  }
  close_attempt()
  assert_eq(events, ["lease"])
  assert_true(lease_open.val)
  assert_true(numeric_open.val)
  fail_lease_close.val = false
  close_attempt()
  assert_eq(events, ["lease", "lease", "numeric"])
  assert_false(lease_open.val)
  assert_false(numeric_open.val)
}
