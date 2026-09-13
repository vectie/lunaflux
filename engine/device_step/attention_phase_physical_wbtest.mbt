///|
/// Opt-in AOT probe of the actual ordered/captured executor construction and
/// scalar selection. Ordinary native tests do not require a CUDA device.
async test "physical mixed attention owns disjoint row writes and unchanged pure graphs" {
  guard @env.get_env_var("LUNAFLUX_ATTENTION_PHASE_FIXTURE") is Some(path) else {
    return
  }
  let context = @device.open_context(0)
  defer (try! context.close())
  let stream = context.create_stream()
  defer (try! stream.close())
  let counts = context.allocate(40L)
  defer (try! counts.close())
  let state = context.allocate(384L)
  defer (try! state.close())
  let loaded = context.load_module(@fs.read_file(path).binary())
  defer (try! loaded.close())
  let ingress = loaded.load_function(b"phase_ingress")
  defer (try! ingress.close())
  let attention = loaded.load_function(b"phase_attention")
  defer (try! attention.close())
  let decode = loaded.load_function(b"phase_decode")
  defer (try! decode.close())
  let output = loaded.load_function(b"phase_output")
  defer (try! output.close())
  let dimensions = @device.KernelLaunchDimensions::new(
    grid_x=1,
    grid_y=1,
    grid_z=1,
    block_x=32,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  let result_argument = @device.KernelArgument::device_region(
    state,
    offset=0L,
    byte_count=384L,
    alignment=4L,
  )
  let arguments = @device.KernelArguments::new([
    @device.KernelArgument::device_region(
      counts,
      offset=0L,
      byte_count=20L,
      alignment=4L,
    ),
    result_argument,
  ])
  let prefill_arguments = @device.KernelArguments::new([
    @device.KernelArgument::device_region(
      counts,
      offset=20L,
      byte_count=20L,
      alignment=4L,
    ),
    result_argument,
  ])
  let resources = empty_paged_resources()
  resources.stream = Some({ resource: stream, open: true, })
  for function in [ingress, attention, decode, output] {
    resources.functions.push({
      resource: function,
      module_index: 0,
      open: true,
    })
  }
  let failure : Ref[DeviceStepError?] = Ref(None)
  defer {
    close_paged_ordered_executors(resources, failure)
    try! assert_eq(failure.val, None)
    try! assert_false(paged_ordered_executor_open(resources))
  }
  let operations : ReadOnlyArray[@plan.OperationId] = [
    @plan.OperationId::new(0),
    @plan.OperationId::new(1),
    @plan.OperationId::new(2),
  ]
  let steps = ReadOnlyArray::from_array(
    [0, 1, 3].mapi((index, function_index) => {
      operation_id: operations[index],
      function_index,
      dimensions,
      arguments,
    }),
  )
  register_attention_phase_replacement(
    resources,
    { ..steps[1], arguments: prefill_arguments, },
    Some({ ..steps[1], function_index: 2, }),
  )
  let launches = paged_ordered_launches(steps, resources)
  for policy in [@device.OrderedKernelEagerOnly, OrderedKernelCaptureRequired] {
    resources.effect_capture_budget = effect_capture_budget(GraphMemoryAbsent)
    let base = prepare_one_paged_ordered_executor(
      context,
      launches[:],
      operations,
      resources,
      policy,
      source_steps=steps,
    )
    let mixed = select_paged_attention_phase_owner(resources, base, MixedGraph)
    assert_true(mixed != base)
    assert_eq(resources.ordered_executors[base].resource.kernel_count(), 3)
    assert_eq(resources.ordered_executors[mixed].resource.kernel_count(), 4)
    assert_eq(
      select_paged_attention_phase_owner(resources, base, PrefillGraph),
      base,
    )
    assert_eq(
      select_paged_attention_phase_owner(resources, base, DecodeGraph),
      base,
    )
    for shape in [(2, 2, 15), (3, 1, 19), (3, 0, 17), (0, 4, 4), (0, 0, 0)] {
      let (prefill, decode_rows, tokens) = shape
      let host = FixedArray::make(40, b'\x00')
      for
        cell, value in [
          prefill,
          decode_rows,
          prefill + decode_rows,
          tokens,
          0,
          prefill,
          0,
          prefill,
          tokens - decode_rows,
          0,
        ] {
        write_i32(host, cell, value)
      }
      counts.copy_from_host(Bytes::from_array(host), 0L, 0L, 40L)
      for owner_index in [base, mixed] {
        let executor = resources.ordered_executors[owner_index].resource
        if executor.execution_mode() == OrderedKernelCaptured {
          executor.launch_captured()
        } else {
          for index in 0..<executor.kernel_count() {
            executor.enqueue(index)
          }
        }
        executor.record_completion()
        executor.wait_completion()
        let result = state.copy_to_host(0L, 384L)
        for token in 0..<tokens {
          assert_eq(result[(64 + token) * 4], b'\x01')
        }
        executor.reset()
      }
    }
  }
  resources.effect_capture_budget = @luna_execution_graph_strategy.ExecutionGraphVariantBudget::new(
    0, 0,
  )
  let budgeted = prepare_one_paged_ordered_executor(
    context,
    launches[:],
    operations,
    resources,
    OrderedKernelCaptureRequired,
    source_steps=steps,
  )
  assert_eq(
    select_paged_attention_phase_owner(resources, budgeted, MixedGraph),
    budgeted,
  )
  println(
    "attention_phase_physical=passed ordered=passed captured=passed mixed_disjoint=passed original_effect_order=passed pure_launch_count=unchanged budget_fallback=preserved",
  )
}

///|
/// Exercise the production variable-length descriptor transaction rather than
/// a fixture-owned full allocation copy. Narrow counts must be visible even
/// when output demand uploads only the shortest row-end suffix.
async test "physical mixed attention counts survive actual descriptor uploads" {
  guard @env.get_env_var("LUNAFLUX_ATTENTION_PHASE_FIXTURE") is Some(_) else {
    return
  }
  let context = @device.open_context(0)
  defer (try! context.close())
  for projected in [false, true] {
    for wire in [false, true] {
      let prepared = prepare_validated(
        context,
        step_test_model_plan(),
        @worker_protocol.ModelPlanGeneration::from_wire(1UL),
        step_test_layout(),
        step_device_limits(max_tokens=32),
        step_required_capabilities(),
        false,
        0UL,
      )
      guard prepared is Ready(descriptor) else {
        fail("descriptor preparation")
      }
      defer (try! descriptor.close())
      descriptor.host.project_output_rows = projected
      for
        plan in [
          attention_phase_ragged_fixture(),
          output_rows_fixture([false, false, false, false], sequence=2UL),
        ] {
        let staged = if wire {
          descriptor.stage_frame(encoded_step_frame(plan))
        } else {
          descriptor.stage(plan)
        }
        let cells = uploaded_count_cells(
          descriptor.host,
          descriptor.staged_summary,
        )
        let copied = FixedArray::from_array(
          descriptor.resources.counts
          .copy_to_host(0L, byte_capacity(cells))
          .to_array(),
        )
        for cell in 0..<cells {
          assert_eq(
            read_descriptor_i32(copied, cell),
            read_descriptor_i32(descriptor.host.counts, cell),
          )
        }
        let summary = descriptor.staged_summary
        for
          cell, value in [
            summary.prefill_count,
            0,
            summary.prefill_count,
            summary.token_count - summary.decode_count,
            summary.page_count,
          ] {
          assert_eq(
            read_descriptor_i32(copied, PREFILL_ATTENTION_COUNTS_CELL + cell),
            value,
          )
        }
        descriptor.finish(staged)
      }
    }
  }
  println(
    "attention_phase_descriptor_upload=passed submitted=passed wire=passed projected=passed unprojected=passed zero_output=passed",
  )
}
