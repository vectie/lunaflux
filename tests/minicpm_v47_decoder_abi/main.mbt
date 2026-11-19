///|
extern "C" fn live_resources() -> Int32 = "lunaflux_device_worker_fake_live_children"

///|
extern "C" fn launches() -> UInt64 = "lunaflux_device_worker_fake_launch_count"

///|
extern "C" fn creates() -> UInt64 = "lunaflux_device_worker_fake_resource_create_count"

///|
extern "C" fn closes() -> UInt64 = "lunaflux_device_worker_fake_resource_close_count"

///|
extern "C" fn copies() -> UInt64 = "lunaflux_device_worker_fake_copy_count"

///|
extern "C" fn reads() -> UInt64 = "lunaflux_device_worker_fake_readback_count"

///|
extern "C" fn set_sample_token(token : Int32) = "lunaflux_device_worker_fake_set_sample_token"

///|
extern "C" fn set_fault(mode : Int32) = "lunaflux_device_worker_fake_set_fault"

///|
extern "C" fn alloc_begin() = "lunaflux_device_worker_alloc_probe_begin"

///|
extern "C" fn alloc_end() -> UInt64 = "lunaflux_device_worker_alloc_probe_end"

///|
extern "C" fn seed() -> Int32 = "lunaflux_device_worker_alloc_probe_seed"

///|
#borrow(value)
extern "C" fn keep_alive(value : FixedArray[Int32]) = "lunaflux_device_worker_alloc_probe_keep_alive"

///|
fn submit(executor : @device.OrderedKernelExecutor) -> Unit raise {
  for index in 0..<executor.kernel_count() {
    executor.enqueue(index)
  }
  executor.record_completion()
  executor.wait_completion()
}

///|
fn measured_submit(
  executor : @device.OrderedKernelExecutor,
  reset~ : Bool = false,
) -> Unit raise {
  let before = (creates(), closes(), copies(), reads())
  alloc_begin()
  if reset {
    executor.reset()
  }
  submit(executor)
  let observed = alloc_end()
  assert_eq(observed, 0UL)
  assert_eq((creates(), closes(), copies(), reads()), before)
}

///|
fn measured_request_step(
  runner : @request.Runner,
  produces_token~ : Bool,
  poll~ : Bool = false,
) -> Unit raise {
  let before_create = creates()
  let before_close = closes()
  let before_copy = copies()
  let before_read = reads()
  alloc_begin()
  let submitted = runner.submit_next()
  let completed = if poll {
    runner.poll()
  } else {
    runner.wait()
    true
  }
  let observed = alloc_end()
  assert_true(submitted && completed)
  assert_eq(observed, 0UL)
  assert_eq(creates(), before_create)
  assert_eq(closes(), before_close)
  assert_eq(copies() - before_copy, 10UL)
  assert_eq(reads() - before_read, if produces_token { 1UL } else { 0UL })
}

///|
fn upload_i32(
  context : @device.Context,
  allocation : @device.Allocation,
  values : ArrayView[Int32],
) -> Unit raise {
  let bytes = FixedArray::makei(values.length() * 4, index => {
    (values[index / 4].reinterpret_as_uint() >> (index % 4 * 8)).to_byte()
  })
  context.copy_from_fixed_host(
    allocation,
    bytes,
    0L,
    0L,
    bytes.length().to_int64(),
  )
}

///|
async fn run_fixture() -> Unit {
  @async.with_task_group() <| group => {
    let spec = @model.Spec::parse(
      @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
    )
    let operators = @operators.DecoderOperators::new(spec, maximum_rows=32)
    let attention = @attention.compile(
      operators,
      device_target=@catalog.DeviceTarget::new(
        compute_major=12,
        compute_minor=1,
        supports_bf16=true,
        supports_cublas_lt=false,
      ),
      target_id=@strategy.AttentionTargetId::new(
        "sm121-unqualified-software-profile",
      ),
      capabilities=@strategy.AttentionCompilerCapabilities::new(
        subgroup_width=32,
        max_threads_per_block=256,
        max_shared_memory_bytes=49152L,
        max_pipeline_stages=1,
        parallel_workgroup_target=1,
        supports_async_copy=false,
        matrix_shape=Some(
          @strategy.AttentionMatrixShape::new(rows=16, columns=16, reduction=16),
        ),
      ),
    )
    let program = @decoder.Program::new(operators, attention)
    let text_program = @text.Program::new(
      operators,
      program,
      maximum_workspace_bytes=1048576L,
    )
    let context = @device.open_context(0)
    group.add_defer(() => context.close())
    let allocations : Array[@device.Allocation] = []
    let modules : Array[@device.Module] = []
    group.add_defer(() => {
      while allocations.last() is Some(allocation) {
        allocation.close()
        ignore(allocations.pop())
      }
    })
    group.add_defer(() => {
      while modules.last() is Some(aot) {
        aot.close()
        ignore(modules.pop())
      }
    })
    let allocate = fn(bytes) raise {
      let allocation = context.allocate(bytes)
      allocations.push(allocation)
      allocation
    }
    let load = fn() raise {
      let aot = context.load_module(b"ABI-only test double; not CUDA code")
      modules.push(aot)
      aot
    }
    let storage = @text.Storage::new(text_program)
    group.add_defer(() => storage.close())
    let before_storage = creates()
    let rejected = @text.Storage::new(text_program)
    let over_budget = try
      rejected.prepare(context, maximum_device_bytes=1L)
    catch {
      _ => true
    } noraise {
      _ => false
    }
    rejected.close()
    assert_true(over_budget)
    assert_eq(creates(), before_storage)
    storage.prepare(
      context,
      maximum_device_bytes=text_program.storage_budget().owned_bytes,
    )
    assert_eq(creates() - before_storage, 22UL)
    let buffers = storage.buffers()
    let states = storage.states()
    let text_metadata = storage.metadata()
    let metadata = text_metadata.decoder
    let sampled_tokens = storage.sampled_tokens()
    let aot : @decoder.Modules = {
      envelope: load(),
      attention: { common: load(), prefill: load(), decode: load(), },
      recurrent: load(),
      moe: load(),
    }
    upload_i32(context, metadata.attention.counts, [1, 0, 1, 1, 1])
    upload_i32(context, metadata.attention.canvas_positions, [0, 0, 0])
    upload_i32(context, metadata.attention.causal_positions, [0])
    upload_i32(context, metadata.attention.row_offsets, [0, 1])
    upload_i32(context, metadata.attention.sequence_lengths, [1])
    upload_i32(context, metadata.attention.page_offsets, [0, 1])
    upload_i32(context, metadata.attention.physical_pages, [0])
    upload_i32(context, metadata.request_slots, [0])
    upload_i32(context, metadata.reset, [1])
    // Immutable synthetic storage is shared to keep the ABI fixture small. The
    // test double validates extents; it does not evaluate these weight bytes.
    let bank = allocate(operators.routed_experts().bank_bytes())
    let ordinary = allocate(33554432L)
    let moe : @moe.Weights = {
      router: ordinary,
      routed_bank: bank,
      shared_gate: ordinary,
      shared_up: ordinary,
      shared_down: ordinary,
      shared_gate_projection: ordinary,
    }
    let recurrent : @recurrent.Weights = {
      qkv: ordinary,
      gate: ordinary,
      decay: ordinary,
      beta: ordinary,
      output: ordinary,
      convolution: ordinary,
      norm: ordinary,
      a_log: ordinary,
      dt_bias: ordinary,
    }
    let full : @attention.Weights = {
      query: ordinary,
      key: ordinary,
      value: ordinary,
      output: ordinary,
      query_norm: ordinary,
      key_norm: ordinary,
    }
    let layers : ReadOnlyArray[@decoder.LayerWeights] = program
      .layers()
      .map(layer => {
        input_norm: ordinary,
        post_attention_norm: ordinary,
        moe,
        branch: match layer.kind {
          GatedDeltaLinearAttention =>
            @decoder.BranchWeights::Recurrent(recurrent)
          FullAttention => FullAttention(full)
        },
      })
    let text_module = load()
    upload_i32(context, text_metadata.token_ids, [10])
    upload_i32(context, text_metadata.output_count, [1])
    upload_i32(context, text_metadata.selected_rows, [0])
    let text_weights : @text.Weights = {
      embedding: allocate(1016397824L),
      final_norm: ordinary,
      language_head: allocate(1016397824L),
    }
    if @env.args()[1:] is ["serve-planner-fixture", ready, proposal] {
      serve_planner_fixture(
        spec,
        text_program,
        context,
        layers[:],
        text_weights,
        ready,
        proposal,
      )
      return
    }
    let frame = @text.Frame::new(text_program)
    group.add_defer(() => frame.close())
    let invalid = @decoder.Frame::new(program)
    let before_invalid = creates()
    let aliased = try
      invalid.prepare(context, modules=aot, metadata~, layers~, states~, buffers={
        ..buffers,
        branch_workspace: buffers.moe_workspace,
      })
    catch {
      _ => true
    } noraise {
      _ => false
    }
    invalid.close()
    assert_true(aliased)
    assert_eq(creates(), before_invalid)
    let resources_before = creates()
    frame.prepare(
      context,
      modules={ decoder: aot, text: text_module, },
      metadata=text_metadata,
      layers~,
      states~,
      buffers~,
      weights=text_weights,
      sampled_tokens~,
    )
    // 776 function handles plus the existing egress owner's two allocations.
    assert_eq(creates() - resources_before, 778UL)
    assert_eq(frame.logits().byte_count(), 992576L)
    assert_eq(frame.launches(Prefill).length(), 924)
    assert_eq(frame.launches(Decode).length(), 924)
    let stream = context.create_stream()
    group.add_defer(() => stream.close())
    let prefill = context.create_ordered_kernel_executor(
      stream,
      frame.launches(Prefill),
      policy=OrderedKernelEagerOnly,
    )
    group.add_defer(() => prefill.close())
    let decode = context.create_ordered_kernel_executor(
      stream,
      frame.launches(Decode),
      policy=OrderedKernelEagerOnly,
    )
    group.add_defer(() => decode.close())
    let before = launches()
    measured_submit(prefill)
    // Input staging belongs to the caller and is outside queue execution scope.
    upload_i32(context, metadata.attention.counts, [0, 1, 1, 1, 1])
    upload_i32(context, metadata.reset, [0])
    for position in 1..<=2 {
      upload_i32(context, text_metadata.token_ids, [10 + position])
      upload_i32(context, metadata.attention.canvas_positions, [
        position, position, position,
      ])
      upload_i32(context, metadata.attention.causal_positions, [position])
      upload_i32(context, metadata.attention.sequence_lengths, [position + 1])
      measured_submit(decode, reset=position == 2)
    }
    assert_eq(launches() - before, 2772UL)
    run_multimodal_fixture(spec, text_program, context, frame, storage)
    // The real request driver stages metadata, retires chunks, feeds generated
    // IDs into decode, reads only producing frames, and owns cancellation.
    let runner = @request.Runner::new(text_program)
    group.add_defer(() => runner.close())
    let before_prepare_copies = copies()
    runner.prepare(context, stream, frame, storage)
    assert_eq(copies() - before_prepare_copies, 2UL)
    let prompt = @prompt.prepare_token_prompt(
      spec,
      Array::makei(33, i => 1000 + i),
      maximum_tokens=4096,
    )
    runner.begin(
      prompt,
      maximum_new_tokens=8,
      stop_token_ids=ReadOnlyArray::from_array(@prompt.stop_token_ids()),
    )
    let before_request = launches()
    set_sample_token(-1)
    measured_request_step(runner, produces_token=false)
    assert_eq(runner.cached_tokens(), 32)
    assert_eq(runner.sampled_tokens(), 0)
    set_sample_token(100)
    measured_request_step(runner, produces_token=true, poll=true)
    assert_eq(runner.cached_tokens(), 33)
    set_sample_token(101)
    measured_request_step(runner, produces_token=true)
    set_sample_token(248046)
    measured_request_step(runner, produces_token=true, poll=true)
    assert_eq(launches() - before_request, 3696UL)
    assert_eq(runner.cached_tokens(), 35)
    assert_eq(runner.sampled_tokens(), 3)
    assert_eq(runner.emitted_tokens(), [100, 101])
    assert_eq(
      runner.finish_reason(),
      Some(@inference.FinishReason::StopSequence),
    )
    assert_true(!runner.submit_next())
    // Cancel an in-flight prefill, then reuse queues/metadata for a new request.
    runner.begin(prompt, maximum_new_tokens=2, stop_token_ids=[])
    assert_true(runner.submit_next())
    runner.cancel()
    assert_eq(runner.finish_reason(), Some(@inference.FinishReason::Cancelled))
    let one = @prompt.prepare_token_prompt(spec, [42], maximum_tokens=4096)
    runner.begin(one, maximum_new_tokens=1, stop_token_ids=[])
    set_sample_token(102)
    measured_request_step(runner, produces_token=true)
    assert_eq(runner.emitted_tokens(), [102])
    assert_eq(runner.finish_reason(), Some(@inference.FinishReason::TokenLimit))
    runner.begin(one, maximum_new_tokens=1, stop_token_ids=[])
    set_sample_token(-1)
    assert_true(runner.submit_next())
    assert_true(
      try runner.wait() catch {
        _ => true
      } noraise {
        _ => false
      },
    )
    assert_eq(runner.cached_tokens(), 0)
    runner.cancel()
    runner.begin(one, maximum_new_tokens=1, stop_token_ids=[])
    set_fault(1)
    assert_true(
      try runner.submit_next() catch {
        _ => true
      } noraise {
        _ => false
      },
    )
    assert_true(
      try runner.begin(one, maximum_new_tokens=1, stop_token_ids=[]) catch {
        _ => true
      } noraise {
        _ => false
      },
    )
    runner.cancel()
    runner.begin(one, maximum_new_tokens=1, stop_token_ids=[])
    set_sample_token(103)
    measured_request_step(runner, produces_token=true)
    assert_eq(runner.emitted_tokens(), [103])
    assert_eq(runner.cached_tokens(), 1)
    run_chat_fixture(spec, runner)
    // Exercise complete execution ownership independently of weight upload.
    let images : @runtime.ModuleImages = {
      envelope: b"ABI envelope",
      recurrent: b"ABI recurrent",
      moe: b"ABI moe",
      attention_common: b"ABI common",
      attention_prefill: b"ABI prefill",
      attention_decode: b"ABI decode",
      text: b"ABI text",
    }
    let payload_budget = text_program.storage_budget().owned_bytes +
      text_program.workspace_bytes()
    let admission = @runtime.Execution::new(text_program)
    let before_admission = creates()
    assert_true(
      try
        admission.prepare(
          context,
          images,
          layers~,
          weights=text_weights,
          maximum_mutable_bytes=payload_budget - 1L,
        )
      catch {
        _ => true
      } noraise {
        _ => false
      },
    )
    assert_eq(creates(), before_admission)
    admission.close()
    let partial = @runtime.Execution::new(text_program)
    let live_before_partial = live_resources()
    assert_true(
      try
        partial.prepare(
          context,
          images,
          layers=[],
          weights=text_weights,
          maximum_mutable_bytes=payload_budget,
        )
      catch {
        _ => true
      } noraise {
        _ => false
      },
    )
    assert_true(live_resources() > live_before_partial)
    partial.close()
    partial.close()
    assert_eq(live_resources(), live_before_partial)
    let execution = @runtime.Execution::new(text_program)
    group.add_defer(() => execution.close())
    let live_before_execution = live_resources()
    execution.prepare(
      context,
      images,
      layers~,
      weights=text_weights,
      maximum_mutable_bytes=payload_budget,
    )
    let session_runner = execution.runner()
    session_runner.begin(
      one,
      maximum_new_tokens=2,
      stop_token_ids=ReadOnlyArray::from_array(@prompt.stop_token_ids()),
    )
    set_sample_token(248046)
    measured_request_step(session_runner, produces_token=true)
    assert_eq(
      session_runner.finish_reason(),
      Some(@inference.FinishReason::StopSequence),
    )
    assert_eq(session_runner.emitted_tokens(), [])
    session_runner.begin(one, maximum_new_tokens=1, stop_token_ids=[])
    set_sample_token(104)
    measured_request_step(session_runner, produces_token=true, poll=true)
    assert_eq(session_runner.emitted_tokens(), [104])
    // A bounded sampled-ID script drives independent producing steps. It is
    // used only by the cross-repository planner fixture, never model arithmetic.
    assert_true(
      set_sample_script(
        FixedArray::makei(2, i => if i == 0 { 105 } else { 248046 }),
      ),
    )
    assert_true(!sample_script_done())
    session_runner.begin(
      one,
      maximum_new_tokens=4,
      stop_token_ids=ReadOnlyArray::from_array(@prompt.stop_token_ids()),
    )
    measured_request_step(session_runner, produces_token=true)
    assert_true(!sample_script_done())
    measured_request_step(session_runner, produces_token=true)
    assert_true(sample_script_done())
    assert_eq(session_runner.emitted_tokens(), [105])
    assert_eq(
      session_runner.finish_reason(),
      Some(@inference.FinishReason::StopSequence),
    )
    set_sample_token(104)
    // Close the complete owner while work is active: queue abort must precede
    // function, buffer, module and stream release, without closing weights.
    session_runner.begin(prompt, maximum_new_tokens=2, stop_token_ids=[])
    assert_true(session_runner.submit_next())
    execution.close()
    execution.close()
    assert_eq(live_resources(), live_before_execution)
    assert_true(
      try execution.runner() catch {
        _ => true
      } noraise {
        _ => false
      },
    )
    run_image_runtime_fixture(
      spec,
      text_program,
      context,
      layers[:],
      text_weights,
    )
  }
}

///|
async fn main {
  alloc_begin()
  let control = FixedArray::make(8, seed())
  keep_alive(control)
  let observed = alloc_end()
  assert_true(observed > 0UL)
  run_fixture()
  if @env.args()[1:] is ["serve-planner-fixture", _, _] {
    assert_eq(live_resources(), 0)
    assert_eq(creates(), closes())
    println(
      "MiniCPM planner fixture server: completed and all native resources closed; scripted tokens only",
    )
    return
  }
  run_vision_fixture()
  assert_eq(live_resources(), 0)
  assert_eq(creates(), closes())
  println(
    "MiniCPM-V 4.7 text ABI: queue/request/runtime startup, allocation-free steps, producing-only readback, cancellation/reuse, partial startup cleanup and active-session close passed; no GPU computation",
  )
}
