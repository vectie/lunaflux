///|
test "fused QKV readonly span binds full-QKV activation and exact adjacency" {
  let plan = blueprint_model_plan(blueprint_metadata())
  let (activation, row_width) = require_fused_qkv_readonly_plan_span(
    plan, 2, 3, 4,
  )
  assert_eq(activation.as_int(), 3)
  assert_eq(row_width, 8)
}

///|
test "fused QKV readonly span rejects stale indices after plan change" {
  let plan = blueprint_model_plan(blueprint_metadata(), extra_projection=true)
  try ignore(require_fused_qkv_readonly_plan_span(plan, 2, 3, 4)) catch {
    MismatchedFusedGroupedOperation(OperationOrder) => ()
    _ => fail("unexpected changed-plan rejection")
  } noraise {
    _ => fail("stale span reached composite admission")
  }
}

///|
test "fused QKV readonly operation identity rejects invalid index" {
  try ignore(fused_qkv_readonly_operation_id(-1)) catch {
    MismatchedFusedGroupedOperation(OperationOrder) => ()
    _ => fail("unexpected operation identity rejection")
  } noraise {
    _ => fail("invalid operation identity reached composite admission")
  }
}

///|
test "fused QKV readonly rejects split-brain attention fallback evidence" {
  let standalone = ("a".repeat(64), "b".repeat(64))
  let readonly_fallback = standalone
  let fused_kv_write = ("c".repeat(64), "d".repeat(64))
  let qualified_kv_write = fused_kv_write
  try
    require_fused_qkv_readonly_attention_fallback_join(
      standalone, readonly_fallback, fused_kv_write, qualified_kv_write,
    )
  catch {
    MismatchedFusedGroupedOperation(StandaloneFallback) => ()
    _ => fail("unexpected split-brain fallback rejection")
  } noraise {
    _ => fail("split-brain attention fallbacks reached composite admission")
  }
}

///|
test "fused QKV readonly production ABI gate accepts only both canary-free ABIs" {
  let production_readonly = @luna_cuda_paged_attention_readonly_aot.RotatedQueryPagedKvReadOnlyProductionRawPointerV2.canonical_name()
  let diagnostic_readonly = @luna_cuda_paged_attention_readonly_aot.RotatedQueryPagedKvReadOnlyRawPointerV1.canonical_name()
  assert_true(
    fused_qkv_readonly_production_abis_match(
      QkvPositionedRopePagedKvWriteProductionAbiV2,
      production_readonly,
    ),
  )
  assert_false(
    fused_qkv_readonly_production_abis_match(
      QkvPositionedRopePagedKvWriteAbiV1,
      production_readonly,
    ),
  )
  assert_false(
    fused_qkv_readonly_production_abis_match(
      QkvPositionedRopePagedKvWriteProductionAbiV2,
      diagnostic_readonly,
    ),
  )
}

///|
test "production QKV readonly replacement preserves exact two-launch order" {
  let replaced = replace_fused_qkv_readonly_span(
    ReadOnlyArray::from_array([0, 1, 2, 3, 4, 5]),
    20,
    40,
    2,
    3,
    4,
  )
  assert_eq(replaced, [0, 1, 20, 40, 5])
  assert_eq(fused_qkv_production_argument_count(), 13)
  assert_eq(readonly_attention_production_argument_count(), 10)
}

///|
test "production QKV readonly second-launch failure aborts ordered execution" {
  let steps = replace_fused_qkv_readonly_span(
    ReadOnlyArray::from_array([0, 1, 2, 3, 4, 5]),
    20,
    40,
    2,
    3,
    4,
  )
  let lifecycle = Ref(GraphStaged)
  let actions : Array[String] = []
  try
    run_paged_ordered_step_range(
      steps,
      lifecycle.val,
      state => lifecycle.val = state,
      (_, operation) => {
        actions.push("enqueue:\{operation}")
        if operation == 40 {
          raise InvalidExecutor(ResourceState)
        }
      },
      () => actions.push("record"),
      () => actions.push("wait"),
      () => actions.push("abort"),
    )
  catch {
    InvalidExecutor(ResourceState) => ()
    _ => fail("unexpected composite ordered-execution failure")
  } noraise {
    _ => fail("failed readonly launch reached completion")
  }
  assert_eq(actions, [
    "enqueue:0", "enqueue:1", "enqueue:20", "enqueue:40", "abort",
  ])
  assert_eq(lifecycle.val, GraphPoisoned)
}

///|
test "production QKV readonly rollback closes functions before modules" {
  let ordered_open = Ref(true)
  let actions : Array[String] = []
  run_paged_ordered_dependency_close(
    () => {
      actions.push("ordered-abort-close")
      ordered_open.val = false
    },
    () => ordered_open.val,
    () => {
      run_paged_close_order(
        ReadOnlyArray::from_array([
          () => {
            actions.push("readonly-function")
            actions.push("qkv-function")
          },
          () => {
            actions.push("readonly-module")
            actions.push("qkv-module")
          },
        ]),
      )
    },
  )
  assert_eq(actions, [
    "ordered-abort-close", "readonly-function", "qkv-function", "readonly-module",
    "qkv-module",
  ])
}

///|
fn fused_qkv_readonly_live_device_join_fixture(
  observed_uuid? : String = "GPU-01234567-89ab-cdef-0123-456789abcdef",
) -> FusedQkvReadOnlyLiveDeviceJoin raise FusedGroupedOperationError {
  fused_qkv_readonly_live_device_join(
    expected_uuid="GPU-01234567-89ab-cdef-0123-456789abcdef",
    expected_pci="00000000:01:00.0",
    expected_name_hex="525458",
    expected_memory_bytes=17175674880L,
    expected_compute_major=12,
    expected_compute_minor=0,
    expected_bf16=true,
    expected_cublas_lt=true,
    observed_uuid~,
    observed_pci="00000000:01:00.0",
    observed_name_hex="525458",
    observed_memory_bytes=17175674880L,
    observed_compute_major=12,
    observed_compute_minor=0,
    observed_bf16=true,
    observed_cublas_lt=true,
  )
}

///|
test "fused QKV readonly composite constructs an exact live-device join" {
  let join = fused_qkv_readonly_live_device_join_fixture()
  assert_true(join.is_exact())
}

///|
test "fused QKV readonly composite rejects live-device substitution" {
  try
    ignore(
      fused_qkv_readonly_live_device_join_fixture(
        observed_uuid="GPU-fedcba98-7654-3210-fedc-ba9876543210",
      ),
    )
  catch {
    MismatchedFusedGroupedOperation(RuntimeDevice) => ()
    _ => fail("unexpected substituted-device rejection")
  } noraise {
    _ => fail("substituted live device reached composite construction")
  }
}
