///|
priv struct FakeFp8CloseState {
  events : Array[String]
  mut ordered_open : Bool
  mut lease_open : Bool
}

///|
fn run_fake_fp8_close(
  state : FakeFp8CloseState,
  fail_ordered? : Bool = false,
  fail_lease? : Bool = false,
) -> Unit {
  run_paged_ordered_dependency_close(
    () => {
      state.events.push("ordered-abort-close")
      if !fail_ordered {
        state.ordered_open = false
      }
    },
    () => state.ordered_open,
    () => {
      run_paged_close_order(
        ReadOnlyArray::from_array([
          () => state.events.push("functions"),
          () => state.events.push("modules"),
          () => state.events.push("kv"),
          () => state.events.push("activation-and-fp8-evidence"),
          () => state.events.push("descriptor"),
          () => {
            run_paged_weight_dependency_close(
              () => {
                state.events.push("weight-lease")
                if !fail_lease {
                  state.lease_open = false
                }
              },
              () => state.lease_open,
              () => state.events.push("fp8-numeric-weight-owner"),
            )
          },
          () => state.events.push("stream"),
        ]),
      )
    },
  )
}

///|
test "FP8 mixed executor teardown preserves every launch dependency" {
  let state : FakeFp8CloseState = {
    events: [],
    ordered_open: true,
    lease_open: true,
  }
  run_fake_fp8_close(state)
  assert_eq(state.events, [
    "ordered-abort-close", "functions", "modules", "kv", "activation-and-fp8-evidence",
    "descriptor", "weight-lease", "fp8-numeric-weight-owner", "stream",
  ])
  assert_false(state.ordered_open)
  assert_false(state.lease_open)
}

///|
test "FP8 teardown retains dependencies until ordered drain can retry" {
  let state : FakeFp8CloseState = {
    events: [],
    ordered_open: true,
    lease_open: true,
  }
  run_fake_fp8_close(state, fail_ordered=true)
  assert_eq(state.events, ["ordered-abort-close"])
  assert_true(state.ordered_open)
  run_fake_fp8_close(state)
  assert_false(state.ordered_open)
  assert_true(state.events.contains("fp8-numeric-weight-owner"))
}

///|
test "FP8 numeric owner remains retained while its allocation lease is open" {
  let state : FakeFp8CloseState = {
    events: [],
    ordered_open: true,
    lease_open: true,
  }
  run_fake_fp8_close(state, fail_lease=true)
  assert_false(state.events.contains("fp8-numeric-weight-owner"))
  assert_true(state.lease_open)
  // The aggregate can retry without replaying an executable launch.
  state.ordered_open = true
  run_fake_fp8_close(state)
  assert_true(state.events.contains("fp8-numeric-weight-owner"))
  assert_false(state.lease_open)
}
