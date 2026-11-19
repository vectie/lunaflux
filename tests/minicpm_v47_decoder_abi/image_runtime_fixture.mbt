///|
fn measured_image_request_step(
  execution : @image_runtime.Execution,
  produces_token~ : Bool,
  poll~ : Bool = false,
) -> Unit raise {
  let before = (creates(), closes(), copies(), reads())
  alloc_begin()
  let submitted = execution.submit_next()
  let completed = if poll {
    execution.poll()
  } else {
    execution.wait()
    true
  }
  let observed = alloc_end()
  assert_true(submitted && completed)
  assert_eq(observed, 0UL)
  assert_eq((creates(), closes()), (before.0, before.1))
  assert_eq(copies() - before.2, 10UL)
  assert_eq(reads() - before.3, if produces_token { 1UL } else { 0UL })
}

///|
async fn run_image_runtime_fixture(
  spec : @model.Spec,
  text : @text.Program,
  context : @device.Context,
  layers : ArrayView[@decoder.LayerWeights],
  text_weights : @text.Weights,
) -> Unit {
  let initial = live_resources()
  let shared = context.allocate(158662656L)
  defer shared.close()
  let vision = @vision.Program::new(
    spec,
    maximum_patches=4096,
    maximum_crop_patches=2048,
  )
  let vision_weights = @vision.Weights::borrow(spec, vision, context, fn(_) {
    return shared
  })
  let program = @multimodal.Program::new(text)
  let images : @image_runtime.ModuleImages = {
    text: {
      envelope: b"fake",
      recurrent: b"fake",
      moe: b"fake",
      attention_common: b"fake",
      attention_prefill: b"fake",
      attention_decode: b"fake",
      text: b"fake",
    },
    vision: b"fake vision",
    image_scatter: b"fake scatter",
  }
  let budget = @image_runtime.Budget::new(
    program,
    vision,
    text_weight_bytes=0L,
    vision_weight_bytes=0L,
  )
  let admission = @image_runtime.Execution::new(program, vision)
  let before_admission = creates()
  assert_true(
    try
      admission.prepare(
        context,
        images,
        layers~,
        text_weights~,
        vision_weights~,
        maximum_mutable_bytes=budget.mutable_bytes - 1L,
      )
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_eq(creates(), before_admission)
  admission.close()
  let partial = @image_runtime.Execution::new(program, vision)
  let before_partial = live_resources()
  assert_true(
    try
      partial.prepare(
        context,
        images,
        layers=[],
        text_weights~,
        vision_weights~,
        maximum_mutable_bytes=budget.mutable_bytes,
      )
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_true(live_resources() > before_partial)
  partial.close()
  partial.close()
  assert_eq(live_resources(), before_partial)
  let execution = @image_runtime.Execution::new(program, vision)
  defer execution.close()
  execution.prepare(
    context,
    images,
    layers~,
    text_weights~,
    vision_weights~,
    maximum_mutable_bytes=budget.mutable_bytes,
  )
  assert_true(execution.is_finished())
  let plan = @processor.Plan::build(spec, height=56, width=56)
  let pixels = plan.prepare_rgb(
    Bytes::from_array(Array::make(56 * 56 * 3, b'\x80'))[:],
    max_bytes=10000000L,
  )
  let layout = @vision_layout.Layout::build(
    spec,
    plan.target_sizes()[:],
    maximum_patches=4096,
    maximum_crop_patches=2048,
  )
  let input = @vision.Input::new(vision, layout, pixels)
  let prompt = @multimodal.prepare_image_tokens(
    spec,
    ([100] + plan.placeholder().tokens()[:].to_owned())[:],
    [plan],
    maximum_tokens=4096,
  )
  let plain = @prompt.prepare_token_prompt(spec, [100], maximum_tokens=4096)
  let before = (launches(), reads(), copies())
  execution.begin_image(prompt, input, maximum_new_tokens=2, stop_token_ids=[])
  assert_eq(launches() - before.0, 318UL)
  assert_eq(copies() - before.2, 13UL)
  assert_true(!execution.is_finished())
  assert_true(
    try
      execution.begin_text(plain, maximum_new_tokens=1, stop_token_ids=[])
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_true(
    try execution.submit_next() catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_true(execution.poll())
  assert_eq(copies() - before.2, 15UL)
  assert_eq(reads(), before.1)
  // A completed vision queue does not permit feature reuse while text is pending.
  assert_true(
    try
      execution.begin_image(prompt, input, maximum_new_tokens=1, stop_token_ids=[])
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  measured_image_request_step(execution, produces_token=false)
  measured_image_request_step(execution, produces_token=false)
  set_sample_token(100)
  measured_image_request_step(execution, produces_token=true)
  set_sample_token(101)
  measured_image_request_step(execution, produces_token=true, poll=true)
  assert_eq(launches() - before.0, 4017UL)
  assert_eq(execution.emitted_tokens(), [100, 101])
  assert_eq(execution.cached_tokens(), 68)
  assert_eq(execution.sampled_tokens(), 2)
  assert_true(execution.is_finished())
  let before_clear = copies()
  execution.begin_text(plain, maximum_new_tokens=1, stop_token_ids=[])
  assert_eq(copies() - before_clear, 1UL)
  measured_image_request_step(execution, produces_token=true)
  execution.begin_image(prompt, input, maximum_new_tokens=1, stop_token_ids=[])
  execution.cancel()
  assert_true(execution.is_finished())
  assert_eq(execution.finish_reason(), Some(Cancelled))
  set_fault(1)
  assert_true(
    try
      execution.begin_image(prompt, input, maximum_new_tokens=1, stop_token_ids=[])
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  execution.cancel()
  execution.begin_image(prompt, input, maximum_new_tokens=1, stop_token_ids=[])
  execution.wait()
  set_fault(1)
  assert_true(
    try execution.submit_next() catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  execution.cancel()
  execution.begin_text(plain, maximum_new_tokens=1, stop_token_ids=[])
  measured_image_request_step(execution, produces_token=true)
  run_media_queue_fixture(spec, vision, execution)
  run_media_chat_fixture(spec, vision, execution)
  // Closing while vision is active must retire it before feature storage release.
  execution.begin_image(prompt, input, maximum_new_tokens=1, stop_token_ids=[])
  execution.close()
  execution.close()
  assert_eq(live_resources(), initial + 1)
  let active_text = @image_runtime.Execution::new(program, vision)
  defer active_text.close()
  active_text.prepare(
    context,
    images,
    layers~,
    text_weights~,
    vision_weights~,
    maximum_mutable_bytes=budget.mutable_bytes,
  )
  active_text.begin_image(prompt, input, maximum_new_tokens=1, stop_token_ids=[])
  active_text.wait()
  assert_true(active_text.submit_next())
  active_text.close()
  active_text.close()
  shared.close()
  assert_eq(live_resources(), initial)
  println(
    "MiniCPM-V 4.7 combined image runtime ABI: nine modules, vision completion before decoder, device-only features, zero-allocation token steps, reuse/cancel/failure/active close passed; scripted driver only",
  )
}
